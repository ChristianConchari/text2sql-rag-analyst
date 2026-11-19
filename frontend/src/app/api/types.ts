/** Aliases over the types generated from docs/openapi.json (`npm run gen:api`). Do not hand-edit the shapes. */
import type { components } from './schema';

type Schemas = components['schemas'];
export type AskRequest = Schemas['AskRequest'];
export type AskResponse = Schemas['AskResponse'];
export type Citation = Schemas['Citation'];
export type TraceStep = Schemas['TraceStep'];
export type Route = AskResponse['route'];
export type Status = AskResponse['status'];
export type Outcome = TraceStep['outcome'];
