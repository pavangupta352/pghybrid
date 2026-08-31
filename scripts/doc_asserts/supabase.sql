-- Does the function the Supabase guide tells people to paste actually work?
--
-- It is a hand-written copy of the query this library generates, which makes it the
-- thing most likely to drift: the generated SQL is covered from every direction, and
-- until now nothing at all ran this. It is also the guide aimed at the largest hosted
-- Postgres audience, so a broken function there is worse than no guide.
--
-- Runs after the guide's own SQL blocks, in the same scratch schema. Every row returned
-- must have `true` in its first column.

-- Two of the twelve demo clauses, plus the one the whole project uses as its example.
-- The vectors are 1536-wide because the guide's signature says vector(1536); only the
-- first two components carry any signal.
insert into documents (title, content, embedding) values
  ('Automatic extension', 'This agreement extends automatically for successive twelve month terms.',
   ('[1,0' || repeat(',0', 1534) || ']')::vector),
  ('Renewal pricing', 'Renewal pricing is subject to change on notice to the customer.',
   ('[0.7,0.7' || repeat(',0', 1534) || ']')::vector),
  ('Termination for convenience', 'Either party may terminate this agreement by giving sixty days written notice.',
   ('[0.9,0.4' || repeat(',0', 1534) || ']')::vector);

-- The function runs at all, and returns the columns it promises.
select count(*) = 3 as ok
from hybrid_search('renewal notice period', ('[1,0' || repeat(',0', 1534) || ']')::vector, 3);

-- The keyword half is doing something: "Renewal pricing" shares two query words and no
-- vector query is aimed at it, so it must carry a text rank.
select bool_or(text_rank is not null) as ok
from hybrid_search('renewal notice period', ('[1,0' || repeat(',0', 1534) || ']')::vector, 3);

-- And the vector half is: the row nearest the query vector must carry a vector rank of 1.
select (select vector_rank from hybrid_search(
          'renewal notice period', ('[0.9,0.4' || repeat(',0', 1534) || ']')::vector, 3
        ) order by vector_rank limit 1) = 1 as ok;

-- The scores are fused rather than one signal: a row found by both signals must score
-- above 1/(k+1), which is the most either signal alone can contribute.
select bool_or(score > 1.0 / 61.0) as ok
from hybrid_search('renewal notice period', ('[0.9,0.4' || repeat(',0', 1534) || ']')::vector, 3)
where vector_rank is not null and text_rank is not null;

-- A query of nothing but stop words must not error and must not return the whole table
-- ranked by nothing: to_tsvector drops them, string_agg returns NULL, the tsquery is NULL
-- and no row matches @@, so the text half contributes nothing and the vector half answers.
select count(*) = 3 and bool_and(text_rank is null) as ok
from hybrid_search('the and of', ('[1,0' || repeat(',0', 1534) || ']')::vector, 3);
