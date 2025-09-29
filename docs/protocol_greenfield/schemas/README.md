# Schema Directory

JSON Schema definitions for each protocol frame type (`type` field) live here.
Schemas should be versioned using the file naming convention
`<type>.v<version>.schema.json` (e.g. `session.hello.v1.schema.json`).

CI and local tooling should validate both server- and client-emitted payloads
against these schemas.
