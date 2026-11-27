import type { Citation } from '../api/types';

/** A piece of a docs answer: plain text, or a reference to `citations[index]`. */
export type Segment = { text: string } | { cite: number; label: string };

/** "Ley 19.886 art. 19", "ley 19886 artículo 19" -> "ley 19886 art 19" (same tolerance as src/rag/citations.py). */
export function citationKey(s: string): string {
  return s
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/\bart[ií]?culo\b/g, 'art')
    .replace(/\./g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Split an answer into text and citation chips. A bracket becomes chips only if every reference in it is one of
 * the verified `citations`; anything else stays as text (the backend already rejected invented ones).
 */
export function splitCitations(answer: string, citations: Citation[]): Segment[] {
  const keys = citations.map((c) => citationKey(c.label));
  const out: Segment[] = [];
  let last = 0;
  for (const m of answer.matchAll(/\[([^\]]+)\]/g)) {
    const refs = m[1].split(';').map((r) => r.trim()).filter(Boolean);
    const idx = refs.map((r) => keys.indexOf(citationKey(r)));
    if (!refs.length || idx.some((i) => i < 0)) continue;
    if (m.index! > last) out.push({ text: answer.slice(last, m.index) });
    for (const i of idx) out.push({ cite: i, label: `art. ${citations[i].articulo}` });
    last = m.index! + m[0].length;
  }
  if (last < answer.length) out.push({ text: answer.slice(last) });
  return out;
}

/** "Ley 19.886 art. 19" + "19" -> "Ley 19.886 · Artículo 19". */
export function sourceTitle(c: Citation): string {
  const norma = c.label.replace(/\s+art\.\s.*$/i, '');
  return `${norma} · Artículo ${c.articulo}`;
}
