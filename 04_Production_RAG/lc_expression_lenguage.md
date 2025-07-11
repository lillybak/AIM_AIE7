### Imperative vs Declarative flow
─────────────────────────────
 Imperative (step-by-step)
─────────────────────────────
question
   │
   ▼
retriever.invoke(question)
   │
   ▼
docs (context chunks)
   │
   ▼
prompt.format(context=docs, question=question)
   │
   ▼
LLM(prompt text)
   │
   ▼
parse(output)
   │
   ▼
final_answer
─────────────────────────────
Explicit: "how" steps coded manually
─────────────────────────────


─────────────────────────────
 LCEL declarative flow
─────────────────────────────
question
   │
   ▼
{ "context": retriever, "question": RunnablePassthrough() }
   │
   ▼
prompt
   │
   ▼
LLM
   │
   ▼
StrOutputParser()
   │
   ▼
final_answer
─────────────────────────────
Declarative: "what" you want described as a pipeline
─────────────────────────────

======================================================================================

──────────────────────────────────────────────
 DIRECT RETRIEVER APPROACH
──────────────────────────────────────────────

           "What is the loan repayment period?"
                        │
                        ▼
         retriever.invoke(question string)
                        │
                        ▼
              [Context documents]
                        │
                        ▼
           (Manually pass to LLM or inspect)
──────────────────────────────────────────────
 PROS: Simple, quick for testing
 CONS: Not modular, harder to chain

──────────────────────────────────────────────
 STATE-BASED PIPELINE APPROACH
──────────────────────────────────────────────

         State = {"question": "What is the loan repayment period?"}
                        │
                        ▼
                   retrieve()
                  ┌─────────────┐
                  │ Extracts    │
                  │ context     │
                  │ docs using  │
                  │ retriever   │
                  └─────────────┘
                        │
             Adds to state: {"context": [...]}
                        │
                        ▼
                   answer()
                  ┌─────────────┐
                  │ Uses        │
                  │ context and │
                  │ question to │
                  │ call LLM    │
                  └─────────────┘
                        │
              Adds to state: {"answer": "10 years ..."}
                        │
                        ▼
                   final state
──────────────────────────────────────────────
 PROS: Modular, reusable steps, easy to extend
 CONS: Slightly more verbose initially

