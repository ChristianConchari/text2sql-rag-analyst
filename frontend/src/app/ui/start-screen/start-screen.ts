import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

import { ROUTE_LABEL } from '../../core/format';
import { Composer } from '../composer/composer';

export const START_CARDS = [
  { route: 'sql', question: '¿Cuántas órdenes de compra se enviaron en enero de 2026?' },
  { route: 'sql', question: '¿Cuántos organismos públicos distintos emitieron órdenes de compra?' },
  { route: 'docs', question: '¿Cuándo procede comprar por cotización?' },
  { route: 'docs', question: '¿Pueden contratar con el Estado empresas o personas extranjeras?' },
] as const;

@Component({
  selector: 'app-start-screen',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Composer],
  templateUrl: './start-screen.html',
  styleUrl: './start-screen.scss',
})
export class StartScreen {
  readonly loading = input(false);
  readonly ask = output<string>();
  protected readonly cards = START_CARDS;
  protected readonly label = ROUTE_LABEL;
}
