import { cell, seconds } from './format';

describe('format', () => {
  it('seconds with one decimal and a comma', () => {
    expect(seconds(412)).toBe('0,4 s');
    expect(seconds(0)).toBe('0,0 s');
    expect(seconds(11168.2)).toBe('11,2 s');
  });

  it('numbers in es-CL, text as is, null empty', () => {
    expect(cell(131720)).toBe('131.720');
    expect(cell(1234.5)).toBe('1.234,5');
    expect(cell('2026-01-05')).toBe('2026-01-05');
    expect(cell(null)).toBe('');
  });
});
