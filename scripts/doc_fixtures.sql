-- Tables the documentation's code blocks refer to.
--
-- The guides address people who already have a table, so their examples name real-looking
-- ones -- `documents` for the Neon guide, pgai's `blog_contents_embeddings` store for the
-- migration guide. Creating them here is what lets those blocks be run rather than only
-- read, without putting setup nobody needs into a page someone is reading to migrate.
--
-- Used by scripts/check_readme_code.py. Safe to run repeatedly.

DROP TABLE IF EXISTS documents;
CREATE TABLE documents (
    id        bigserial PRIMARY KEY,
    content   text NOT NULL,
    embedding vector(8),
    fts       tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED
);
INSERT INTO documents (content, embedding) VALUES
    ('the renewal notice period is sixty days', '[1,0,0,0,0,0,0,0]'),
    ('termination for convenience clause',      '[0.9,0.1,0,0,0,0,0,0]');

-- pgai's destination table: the text column is `chunk`, which is also what the guide
-- claims `pghybrid init` finds on its own.
DROP TABLE IF EXISTS blog_contents_embeddings;
CREATE TABLE blog_contents_embeddings (
    id        bigserial PRIMARY KEY,
    chunk     text NOT NULL,
    embedding vector(8),
    fts       tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(chunk, ''))) STORED
);
INSERT INTO blog_contents_embeddings (chunk, embedding) VALUES
    ('the renewal notice period is sixty days', '[1,0,0,0,0,0,0,0]'),
    ('termination for convenience clause',      '[0.9,0.1,0,0,0,0,0,0]');
