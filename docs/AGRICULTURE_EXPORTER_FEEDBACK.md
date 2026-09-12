# Agriculture & Commodity: exporter feedback sheet

Use this sheet with a real exporter or a representative agriculture-sector user
after the internal demonstration is complete. It is a discussion guide, not a
record of feedback that has already been collected.

## Before the session

- Obtain the participant's agreement to take part and explain that the system
  is a student-project prototype, not financial, trade, or purchasing advice.
- Do not store names, company names, contact details, or commercially sensitive
  figures in this repository.
- Use the local evaluation environment and state that the answers may be
  degraded when no LLM is configured.
- Record only anonymised role information, for example `tea exporter` or
  `agriculture analyst`, and only if the participant agrees.

## Suggested session flow (10--15 minutes)

1. Explain the source/evidence and limitation approach in one minute.
2. Ask the participant to read the cinnamon forecast result.
3. Show the Tea Board/UN Comtrade `dq_flag` example.
4. Ask the questions below and record anonymised responses.
5. Do not promise that any suggestion will be implemented.

## Questions

Rate each closed question from 1 (strongly disagree) to 5 (strongly agree).

| ID | Prompt | Response |
|---|---|---|
| F1 | I can understand what the cinnamon forecast means. | 1 2 3 4 5 |
| F2 | The 80% interval and the 33% back-test coverage limitation are clear. | 1 2 3 4 5 |
| F3 | Showing the Tea Board and UN Comtrade discrepancy helps me judge the tea figure. | 1 2 3 4 5 |
| F4 | The stated data limitations make me trust the system more than an unsupported answer would. | 1 2 3 4 5 |
| F5 | I could use this type of evidence when exploring an export decision. | 1 2 3 4 5 |

Open questions:

1. Which figure, source, or limitation was unclear?
2. Which agriculture or commodity question would be most useful in your work?
3. What evidence would you need before acting on a forecast?
4. Did the `dq_flag` explanation help, confuse, or concern you? Why?
5. What should CeyNex refuse to answer unless it has stronger data?

## Anonymised observation record

```text
Session ID:
Date:
Participant role (optional):
Environment: local direct Agriculture Agent / LLM degraded
Questions demonstrated: cinnamon forecast; tea export-volume trend; tea dq_flag

F1: __  F2: __  F3: __  F4: __  F5: __
Most useful capability:
Unclear or misleading point:
Requested improvement:
Evidence/limitation concern:
Consent to quote anonymised feedback in report: yes / no
```

## Reporting rule

Do not state that exporter testing occurred until at least one real participant
has completed a session and has agreed to the way their anonymised feedback will
be reported. If no sessions occur, report that user feedback remains pending;
do not replace it with developer or team opinions.
