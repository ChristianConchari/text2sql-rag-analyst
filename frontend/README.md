# Frontend

Angular 22 (standalone components, signals, zoneless), Angular Material (M3 theme mapped to the design tokens),
Vitest. The design reference is `../docs/ui-design/`.

```bash
nvm use                      # Node 24.22.3 (.nvmrc); Angular 22 needs >= 24.15
npm ci
npm start                    # http://localhost:4200, /api proxied to http://127.0.0.1:8000 (proxy.conf.json)
npm test                     # Vitest, single run
npm run build                # dist/frontend/browser, served by FastAPI at "/" (see ../app/main.py)
npm run gen:api              # regenerate src/app/api/schema.d.ts from the API (writes ../docs/openapi.json first)
```

Layout:

| Path | What |
|---|---|
| `src/app/api/` | `schema.d.ts` generated from `docs/openapi.json` (never edit by hand) and the aliases in `types.ts` |
| `src/app/core/ask-facade.ts` | the only state: status (`idle \| loading \| done \| error`) and the in-memory conversation |
| `src/app/core/theme.ts` | light/dark: follows the system, the header toggle overrides it (localStorage `theme`) |
| `src/app/core/format.ts`, `citations.ts` | es-CL numbers, durations ("0,4 s"), route labels; citation chips |
| `src/app/ui/` | start screen, composer, pending (loading), answer, trace panel |

No streaming yet: while a request runs the UI shows elapsed time and generic progress; the trace appears with the
answer. The conversation is not persisted.
