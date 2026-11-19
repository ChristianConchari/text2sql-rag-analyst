import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { DOCS_OK, SQL_OK } from '../../testing/fixtures';
import { AskFacade, toAskError } from './ask-facade';

describe('AskFacade', () => {
  let facade: AskFacade;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    facade = TestBed.inject(AskFacade);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('starts idle with an empty conversation', () => {
    expect(facade.status()).toBe('idle');
    expect(facade.empty()).toBe(true);
  });

  it('goes idle -> loading -> done and stores the answer', () => {
    facade.ask('  ¿Cuántas órdenes?  ');
    expect(facade.status()).toBe('loading');
    expect(facade.turns()[0].question).toBe('¿Cuántas órdenes?');

    const req = http.expectOne('/api/ask');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ question: '¿Cuántas órdenes?' });
    req.flush(SQL_OK);

    expect(facade.status()).toBe('done');
    expect(facade.turns()[0].response).toEqual(SQL_OK);
    expect(facade.model()).toBe('qwen2.5-coder:14b-8k');
  });

  it('ignores blank questions and a second question while loading', () => {
    facade.ask('   ');
    http.expectNone('/api/ask');
    facade.ask('uno');
    facade.ask('dos');
    http.expectOne('/api/ask').flush(SQL_OK);
    expect(facade.turns().map((t) => t.question)).toEqual(['uno']);
  });

  it('keeps the conversation across questions', () => {
    facade.ask('uno');
    http.expectOne('/api/ask').flush(SQL_OK);
    facade.ask('dos');
    http.expectOne('/api/ask').flush(DOCS_OK);
    expect(facade.turns().map((t) => t.response?.route)).toEqual(['sql', 'docs']);
  });

  it('goes to error on a 500 and can retry', () => {
    facade.ask('uno');
    http.expectOne('/api/ask').flush('boom', { status: 500, statusText: 'Internal Server Error' });
    expect(facade.status()).toBe('error');
    const failed = facade.turns()[0];
    expect(failed.error).toMatchObject({ kind: 'server', httpStatus: 500 });

    facade.retry(failed.id);
    expect(facade.status()).toBe('loading');
    http.expectOne('/api/ask').flush(SQL_OK);
    expect(facade.turns()).toHaveLength(1);
    expect(facade.status()).toBe('done');
  });

  it('reports the API down (network error) and marks it offline', () => {
    facade.ask('uno');
    http.expectOne('/api/ask').error(new ProgressEvent('error'), { status: 0 });
    expect(facade.status()).toBe('error');
    expect(facade.turns()[0].error?.kind).toBe('network');
    expect(facade.online()).toBe(false);
  });

  it('maps 422 to an invalid-question error', () => {
    facade.ask('uno');
    http.expectOne('/api/ask').flush({ detail: [] }, { status: 422, statusText: 'Unprocessable Entity' });
    expect(facade.turns()[0].error?.kind).toBe('invalid');
  });

  it('cancel stops waiting and frees the composer', () => {
    facade.ask('uno');
    const req = http.expectOne('/api/ask');
    facade.cancel();
    expect(req.cancelled).toBe(true);
    expect(facade.loading()).toBe(false);
    expect(facade.turns()[0].error?.kind).toBe('cancelled');
  });

  it('reset clears the conversation', () => {
    facade.ask('uno');
    http.expectOne('/api/ask').flush(SQL_OK);
    facade.reset();
    expect(facade.empty()).toBe(true);
    expect(facade.status()).toBe('idle');
  });

  it('health check sets online', () => {
    facade.checkHealth();
    http.expectOne('/api/health').flush({ status: 'ok' });
    expect(facade.online()).toBe(true);
  });

  it('maps proxy gateway errors to network', () => {
    expect(toAskError(new HttpErrorResponse({ status: 504 })).kind).toBe('network');
  });
});
