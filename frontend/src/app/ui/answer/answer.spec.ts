import { ComponentFixture, TestBed } from '@angular/core/testing';

import type { AskResponse } from '../../api/types';
import { DOCS_OK, OUT_OF_SCOPE, REFUSED, SQL_OK } from '../../../testing/fixtures';
import { Answer, SUGGESTIONS } from './answer';

async function render(response: AskResponse): Promise<{ fixture: ComponentFixture<Answer>; el: HTMLElement }> {
  const fixture = TestBed.createComponent(Answer);
  fixture.componentRef.setInput('response', response);
  fixture.componentRef.setInput('turnId', 1);
  await fixture.whenStable();
  return { fixture, el: fixture.nativeElement as HTMLElement };
}

const text = (el: Element | null) => (el?.textContent ?? '').replace(/\s+/g, ' ').trim();

describe('Answer', () => {
  beforeEach(() => TestBed.configureTestingModule({ imports: [Answer] }));

  it('sql: chip, meta, trace, SQL block with Copiar, es-CL table with masked cell', async () => {
    const { el } = await render(SQL_OK);
    expect(text(el.querySelector('.chip'))).toBe('Datos · sql');
    expect(text(el.querySelector('.meta'))).toBe('ok · 2 intentos · 3,1 s');
    expect(text(el.querySelector('.summary'))).toBe('7 pasos · 3,1 s');
    expect(el.querySelectorAll('app-trace-panel li')).toHaveLength(7);
    expect(text(el.querySelector('pre'))).toContain('COUNT(DISTINCT Codigo)');
    expect([...el.querySelectorAll('.code-head button')].map(text)).toContain('Copiar');
    const cells = [...el.querySelectorAll('td')];
    expect(text(cells[0])).toBe('131.720');
    expect(cells[1].classList).toContain('masked');
    expect(el.querySelector('.sources')).toBeNull();
  });

  it('trace: durations in seconds with a comma, objection dot in red, toggle with aria-expanded', async () => {
    const { fixture, el } = await render(SQL_OK);
    const times = [...el.querySelectorAll('.time')].map(text);
    expect(times.slice(0, 3)).toEqual(['0,4 s', '0,9 s', '0,1 s']);
    const dots = [...el.querySelectorAll<HTMLElement>('.dot')];
    expect(dots[2].style.background).toBe('var(--ref)');
    const toggle = el.querySelector<HTMLButtonElement>('.toggle')!;
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    toggle.click();
    await fixture.whenStable();
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(el.querySelector('ol')!.hidden).toBe(true);
  });

  it('docs: citation chip linked to its source card, no SQL or table', async () => {
    const { el } = await render(DOCS_OK);
    expect(text(el.querySelector('.chip'))).toBe('Ley y reglamento · docs');
    const cite = el.querySelector<HTMLAnchorElement>('a.cite')!;
    expect(text(cite)).toBe('1 · art. 19');
    expect(cite.getAttribute('href')).toBe('#fuente-1-1');
    expect(text(el.querySelector('.answer-text'))).not.toContain('[Ley 19.886 art. 19]');
    const source = el.querySelector('#fuente-1-1')!;
    expect(text(source)).toContain('Ley 19.886 · Artículo 19');
    expect(text(source)).toContain('Créase un Sistema');
    expect(el.querySelector('pre')).toBeNull();
    expect(el.querySelector('table')).toBeNull();
  });

  it('out_of_scope: the scope message and two suggested questions that emit', async () => {
    const { fixture, el } = await render(OUT_OF_SCOPE);
    expect(text(el.querySelector('.chip'))).toBe('Fuera de alcance');
    expect(text(el.querySelector('.meta'))).toBe('sin herramientas · 0,4 s');
    expect(text(el.querySelector('.answer-text'))).toContain('Solo puedo responder');
    const buttons = [...el.querySelectorAll<HTMLButtonElement>('.suggestion')];
    expect(buttons).toHaveLength(2);
    const emitted: string[] = [];
    fixture.componentInstance.suggest.subscribe((q) => emitted.push(q));
    buttons[0].click();
    expect(emitted).toEqual([SUGGESTIONS[0].question]);
  });

  it('refuse: the generic message in the refusal box, nothing else', async () => {
    const { el } = await render(REFUSED);
    expect(text(el.querySelector('.chip'))).toBe('Rechazado');
    expect(text(el.querySelector('.refuse'))).toBe('No puedo ayudar con esa solicitud.');
    expect(el.querySelector('pre')).toBeNull();
    expect(el.querySelector('.suggestion')).toBeNull();
    expect(el.querySelectorAll('app-trace-panel li')).toHaveLength(2);
  });
});
