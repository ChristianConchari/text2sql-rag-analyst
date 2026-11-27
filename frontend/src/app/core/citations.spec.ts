import { DOCS_OK } from '../../testing/fixtures';
import { citationKey, splitCitations } from './citations';

describe('citations', () => {
  it('normalizes the accepted spellings', () => {
    expect(citationKey('Ley 19.886 art. 8 bis')).toBe(citationKey('ley 19886 artículo 8 bis'));
  });

  it('turns grouped references into chips and leaves unknown brackets as text', () => {
    const cits = [...DOCS_OK.citations, { label: 'Decreto 661 art. 59', norma: 'Decreto 661', articulo: '59', excerpt: '' }];
    const segs = splitCitations('A [Ley 19.886 art. 19; Decreto 661 art. 59] y [ver tabla].', cits);
    expect(segs).toEqual([
      { text: 'A ' }, { cite: 0, label: 'art. 19' }, { cite: 1, label: 'art. 59' }, { text: ' y [ver tabla].' },
    ]);
  });
});
