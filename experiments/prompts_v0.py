"""Probe prompts: 5 task families x 5 paraphrase groups x 3 paraphrases.

Every group's paraphrases mean the same thing, so a semantically deterministic system must
treat them identically. Decide groups also carry the expected label set for label scoring.
"""

FAMILIES: dict[str, list[list[str]]] = {
    "decide": [
        [
            "A customer has a credit score of 580 and a debt-to-income ratio of 55%. Should the loan be approved? Answer yes or no.",
            "Credit score: 580. Debt-to-income: 55%. Approve the loan? Reply with yes or no.",
            "Would you approve a loan for an applicant whose credit score is 580 and whose debt-to-income ratio is 55%? Answer only yes or no.",
        ],
        [
            "Our policy allows refunds within 30 days of purchase. The customer bought the item 45 days ago. Should we issue a refund? Answer yes or no.",
            "Refunds are allowed for 30 days after purchase. This purchase was 45 days ago. Refund it? Answer yes or no.",
            "The item was purchased 45 days ago and our refund window is 30 days. Do we give a refund? Answer with yes or no.",
        ],
        [
            "Is 91 a prime number? Answer yes or no.",
            "Answer yes or no: is the number 91 prime?",
            "Is ninety-one prime? Reply yes or no.",
        ],
        [
            "Is this email spam: 'Congratulations! You won a free cruise, click here to claim your prize now'? Answer yes or no.",
            "Email text: 'Congratulations! You won a free cruise, click here to claim your prize now'. Is it spam? Yes or no.",
            "Would you classify the message 'Congratulations! You won a free cruise, click here to claim your prize now' as spam? Answer yes or no.",
        ],
        [
            "The server CPU has been at 99% for two hours and response times have tripled. Should we page the on-call engineer? Answer yes or no.",
            "CPU at 99% for 2 hours, latency up 3x. Page on-call? Yes or no.",
            "Response times are three times higher and the CPU has been pegged at 99% for two hours. Do we wake up the on-call engineer? Answer yes or no.",
        ],
    ],
    "call": [
        [
            "Get the weather forecast for Paris for tomorrow.",
            "What will the weather be in Paris tomorrow? Use the weather tool.",
            "Call the weather API: city Paris, day tomorrow.",
        ],
        [
            "Send an email to bob@example.com with the subject 'Meeting moved' saying the meeting is now at 3pm.",
            "Email bob@example.com, subject 'Meeting moved', body: the meeting is now at 3pm.",
            "Write to bob@example.com that the meeting moved to 3pm; use the subject line 'Meeting moved'.",
        ],
        [
            "Convert 100 US dollars to euros.",
            "How much is 100 USD in EUR? Use the currency converter.",
            "Currency conversion: amount 100, from USD, to EUR.",
        ],
        [
            "Create a calendar event called 'Design review' on Friday at 3pm for 30 minutes.",
            "Schedule 'Design review' this Friday, 3:00 pm, duration 30 minutes.",
            "Add a 30-minute event titled 'Design review' to my calendar on Friday at 15:00.",
        ],
        [
            "Search for flights from SFO to JFK on November 2, 2026.",
            "Find flights San Francisco (SFO) to New York (JFK), departing 2026-11-02.",
            "Look up SFO to JFK flights for the 2nd of November 2026.",
        ],
    ],
    "judge": [
        [
            "Question: What is the capital of Australia? Answer A: Sydney. Answer B: Canberra. Which answer is better? Reply A or B.",
            "Which response correctly answers 'What is the capital of Australia?' A) Sydney B) Canberra. Answer A or B.",
            "Compare two answers to 'capital of Australia': (A) Sydney, (B) Canberra. Which is better, A or B?",
        ],
        [
            "Rate the correctness of this Python function from 1 to 5: def add(a, b): return a - b",
            "On a 1-5 scale, how correct is this code? def add(a, b): return a - b",
            "Score this implementation of addition for correctness (1 = wrong, 5 = correct): def add(a, b): return a - b",
        ],
        [
            "Which translation of 'Je suis fatigué' is more accurate? A: 'I am tired.' B: 'I am hungry.' Answer A or B.",
            "French: 'Je suis fatigué'. Translation A: 'I am tired.' Translation B: 'I am hungry.' Which is more accurate, A or B?",
            "Pick the better English translation of 'Je suis fatigué': (A) I am tired (B) I am hungry. Reply with A or B.",
        ],
        [
            "Is this response helpful to the question 'How do I boil an egg?': 'Eggs are a good source of protein.' Answer yes or no.",
            "Question: How do I boil an egg? Response: Eggs are a good source of protein. Is the response helpful? Yes or no.",
            "Does the reply 'Eggs are a good source of protein.' helpfully answer 'How do I boil an egg?' Answer yes or no.",
        ],
        [
            "Which summary is more faithful to 'The meeting was postponed to Monday due to illness'? A: 'The meeting was cancelled.' B: 'The meeting moved to Monday because someone was sick.' Answer A or B.",
            "Source: 'The meeting was postponed to Monday due to illness'. Summary A: 'The meeting was cancelled.' Summary B: 'The meeting moved to Monday because someone was sick.' Which is more faithful, A or B?",
            "For the text 'The meeting was postponed to Monday due to illness', is summary (A) 'The meeting was cancelled.' or (B) 'The meeting moved to Monday because someone was sick.' more faithful? Answer A or B.",
        ],
    ],
    "create": [
        [
            "Write a short poem about the sea.",
            "Compose a brief poem on the theme of the ocean.",
            "Give me a few lines of poetry about the sea.",
        ],
        [
            "Write a product description for a bamboo toothbrush.",
            "Create marketing copy describing a bamboo toothbrush.",
            "Describe a bamboo toothbrush for an online store listing.",
        ],
        [
            "Write the opening paragraph of a story about a lighthouse keeper.",
            "Begin a short story whose main character keeps a lighthouse.",
            "Draft the first paragraph of a tale about a lighthouse keeper.",
        ],
        [
            "Come up with a slogan for a family bakery.",
            "Write a catchy tagline for a small family-run bakery.",
            "Suggest a slogan a family bakery could use.",
        ],
        [
            "Write a haiku about autumn.",
            "Compose a haiku on the subject of autumn.",
            "Give me a haiku about the fall season.",
        ],
    ],
    "reason": [
        [
            "A train travels 120 km in 1.5 hours. What is its average speed in km/h?",
            "What is the average speed in km per hour of a train that covers 120 kilometers in an hour and a half?",
            "Distance 120 km, time 1.5 h. Compute the average speed in km/h.",
        ],
        [
            "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
            "All bloops are razzies. All razzies are lazzies. Does it follow that all bloops are lazzies?",
            "Given that every bloop is a razzie and every razzie is a lazzie, is every bloop a lazzie?",
        ],
        [
            "What is the sum of 14, 27, 39 and 20?",
            "Add up the numbers 14, 27, 39, and 20.",
            "Compute 14 + 27 + 39 + 20.",
        ],
        [
            "What is 17 multiplied by 23?",
            "Compute the product of 17 and 23.",
            "What does 17 x 23 equal?",
        ],
        [
            "Monthly sales were 120 in January, 135 in February and 150 in March. What was the average month-over-month increase?",
            "Sales: Jan 120, Feb 135, Mar 150. What is the average monthly increase?",
            "If sales went from 120 to 135 to 150 over three months, by how much did they increase per month on average?",
        ],
    ],
}

DECIDE_LABELS = ("yes", "no")


def labels_for(text: str) -> tuple[str, ...]:
    """Closed label set implied by a decide/judge prompt."""
    if "1 to 5" in text or "1-5" in text or "1 = wrong" in text:
        return ("1", "2", "3", "4", "5")
    if "A or B" in text:
        return ("A", "B")
    return ("yes", "no")


def decision_ops():
    """Decide-profile IR ops for every decide and judge prompt (30 decisions)."""
    from dmc.ir import Context, DecideClassify

    ctx = Context(now=None, locale="en-US", timezone=None, session_state_hash=None)
    return [
        DecideClassify(context=ctx, ambiguities=(), question=text, labels=labels_for(text),
                       policy=None, evidence=())
        for fam in ("decide", "judge")
        for paras in FAMILIES[fam]
        for text in paras
    ]
