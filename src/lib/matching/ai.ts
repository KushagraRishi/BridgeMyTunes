import "server-only";
import Anthropic from "@anthropic-ai/sdk";
import { env } from "../env";
import type { ScoredCandidate, Track } from "../types";

export interface AiCase {
  key: number;
  source: Track;
  candidates: ScoredCandidate[];
}

export interface AiVerdict {
  key: number;
  /** Index into that case's candidates, or -1 when none is the same recording. */
  choice: number;
  confidence: number;
  differences: string[];
  reason: string;
}

let client: Anthropic | null = null;

type Schema = Record<string, unknown>;

/**
 * One Claude call that must answer with JSON matching `schema`. The answer is wrapped as {"results": [...]}
 * because structured outputs need an object at the top. On a policy decline the API retries on a fallback model
 * by itself ("fallbacks": "default"). Throws when there is no usable answer; callers catch and return [].
 */
async function ask(system: string, prompt: string, item: Schema): Promise<unknown[]> {
  client ??= new Anthropic({ apiKey: env.anthropicApiKey!, timeout: 60_000, maxRetries: 2 });
  const res = await client.beta.messages.create({
    model: env.claudeModel,
    max_tokens: 16000,
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
    output_config: {
      effort: "low",
      format: {
        type: "json_schema",
        schema: {
          type: "object",
          properties: { results: { type: "array", items: item } },
          required: ["results"],
          additionalProperties: false,
        },
      },
    },
    system,
    messages: [{ role: "user", content: prompt }],
  });
  if (res.stop_reason === "refusal") throw new Error(`Claude declined: ${res.stop_details?.category ?? "no category"}`);
  if (res.stop_reason === "max_tokens") throw new Error("Claude ran out of output tokens");
  const text = res.content.map((b) => (b.type === "text" ? b.text : "")).join("");
  const parsed = JSON.parse(text) as { results?: unknown };
  return Array.isArray(parsed.results) ? parsed.results : [];
}

const obj = (properties: Schema, required = Object.keys(properties)): Schema => ({
  type: "object",
  properties,
  required,
  additionalProperties: false,
});
const str = { type: "string" };
const int = { type: "integer" };

export const aiEnabled = () => Boolean(env.anthropicApiKey);

const fmt = (t: Track) =>
  JSON.stringify({
    title: t.title,
    artists: t.artists,
    album: t.album,
    duration_s: t.durationMs ? Math.round(t.durationMs / 1000) : undefined,
    isrc: t.isrc,
    explicit: t.explicit,
  });

const SYSTEM = `You match songs between music streaming catalogs (Spotify, Apple Music).
For each case, decide which candidate is the SAME RECORDING as the source track, or -1 if none is.

Rules:
- Same recording means same performance. A remaster of the same recording is acceptable but note it.
- Live, acoustic, remix, sped-up, slowed, karaoke, cover/tribute, instrumental and demo versions are DIFFERENT recordings unless the source is the same kind of version.
- A version with an extra or missing featured artist is usually a different version (e.g. "Levitating" vs "Levitating (feat. DaBaby)").
- Platforms format names differently: "feat." in the title vs in the artist list, "A & B" vs ["A","B"], " - Remastered 2011" vs "(2011 Remaster)", soundtrack subtitles. These are NOT differences.
- Clean vs explicit is the same song but note it; prefer the candidate whose explicitness matches.
- Duration within ~3 seconds is a strong signal of the same recording; >15 seconds apart usually is not.
- confidence is 0..1: how sure you are your choice is right (for -1, how sure you are none match).
- differences: short human-readable notes such as "Remastered version", "Clean version", "Live version". Empty if identical.
- reason: one short sentence.`;

/** Ask Claude to adjudicate uncertain matches, several per request. Returns [] if AI is disabled or fails. */
export async function adjudicate(cases: AiCase[]): Promise<AiVerdict[]> {
  if (!cases.length || !env.anthropicApiKey) return [];

  const prompt = cases
    .map(
      (c) =>
        `Case ${c.key}\nSource: ${fmt(c.source)}\nCandidates:\n${c.candidates.map((x, i) => `  [${i}] ${fmt(x)}`).join("\n")}`,
    )
    .join("\n\n");

  try {
    const parsed = (await ask(SYSTEM, prompt, obj({
      key: int,
      choice: int,
      confidence: { type: "number" },
      differences: { type: "array", items: str },
      reason: str,
    }))) as AiVerdict[];
    const byKey = new Map(cases.map((c) => [c.key, c]));
    return parsed.filter((v) => {
      const c = byKey.get(v.key);
      return c && Number.isInteger(v.choice) && v.choice >= -1 && v.choice < c.candidates.length;
    });
  } catch (err) {
    console.error("[ai] adjudication failed:", err);
    return [];
  }
}

export interface QueryCase {
  key: number;
  source: Track;
}

export interface QuerySuggestion {
  key: number;
  queries: { title: string; artist: string }[];
}

const SUGGEST = `You help find songs that a music catalog search could not find.
For each case you get a song as it appears on one service (often a messy YouTube video title) and the service to search.
Suggest up to 2 better searches, each a clean song title and its main artist, most likely first.

Rules:
- Remove video noise: "Official Video", "Lyrics", "Audio", "HD", "4K", "Full Song", channel names, label names, hashtags, emoji, years.
- Fix the artist when the listed one is a channel, label or uploader (e.g. "T-Series", "Topic", "VEVO") and the real artist is in the title.
- If the title is in a non-Latin script, give the common Latin-script title too, and the other way round.
- Keep the version words that matter: Live, Remix, Acoustic, Slowed, a featured artist.
- Never invent a different song. If you can't tell what the song is, return no queries for that case.`;

/** Ask Claude for cleaner searches for songs the catalog search missed, several per request. [] if AI is off or fails. */
export async function suggestSearches(cases: QueryCase[], destination: string): Promise<QuerySuggestion[]> {
  if (!cases.length || !env.anthropicApiKey) return [];
  const prompt =
    `Service to search: ${destination}\n\n` + cases.map((c) => `Case ${c.key}\nSong: ${fmt(c.source)}`).join("\n\n");
  try {
    const parsed = (await ask(SUGGEST, prompt, obj({ key: int, queries: { type: "array", items: obj({ title: str, artist: str }) } }))) as QuerySuggestion[];
    const keys = new Set(cases.map((c) => c.key));
    return parsed
      .filter((s) => keys.has(s.key) && Array.isArray(s.queries))
      .map((s) => ({ key: s.key, queries: s.queries.filter((q) => q?.title?.trim()).slice(0, 2) }));
  } catch (err) {
    console.error("[ai] search suggestions failed:", err);
    return [];
  }
}

export interface ReplacementIdea {
  key: number;
  songs: { title: string; artist: string; why: string }[];
}

const REPLACE = `A listener is moving music to another streaming service, and some songs are not available there.
For each missing song, suggest up to 3 real, well-known songs a fan of it would accept instead, most fitting first:
another version of the same song (a different recording, a live or acoustic version, a cover by a known artist) when one
likely exists, otherwise a closely related song by the same artist, otherwise a very similar song by a similar artist.

Rules:
- Only real songs that are widely released on major streaming services. Never invent titles.
- Never suggest the exact same recording that was already searched for.
- why: a few words a listener understands, e.g. "Acoustic version", "Same artist, same album", "Similar 2000s Hindi pop".`;

/** Ask Claude for similar songs to offer instead of ones the destination doesn't have. [] if AI is off or fails. */
export async function suggestReplacements(cases: QueryCase[], destination: string): Promise<ReplacementIdea[]> {
  if (!cases.length || !env.anthropicApiKey) return [];
  const prompt =
    `Destination service: ${destination}\n\n` + cases.map((c) => `Case ${c.key}\nMissing song: ${fmt(c.source)}`).join("\n\n");
  try {
    const parsed = (await ask(REPLACE, prompt, obj({ key: int, songs: { type: "array", items: obj({ title: str, artist: str, why: str }) } }))) as ReplacementIdea[];
    const keys = new Set(cases.map((c) => c.key));
    return parsed
      .filter((r) => keys.has(r.key) && Array.isArray(r.songs))
      .map((r) => ({ key: r.key, songs: r.songs.filter((x) => x?.title?.trim() && x?.artist?.trim()).slice(0, 3) }));
  } catch (err) {
    console.error("[ai] replacement suggestions failed:", err);
    return [];
  }
}
