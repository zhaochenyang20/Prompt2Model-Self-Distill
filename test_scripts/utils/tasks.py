class Task:
    def __init__(self, task_instruction, task_name, examples, expected_content: str, optional_list, metric, labels, is_classification, extraction_examples) -> None:
        self.task_instruction = task_instruction
        self.task_name = task_name
        self.examples = examples
        # expected_content 指 verifier 从 generated inut 里面 extract 出来的字段
        self.expected_content = expected_content
        # optional_list 是额外需要 ban 的 string，对于 input output 都会 ban 掉
        self.optional_list = optional_list
        self.metric = metric
        # 对于分类任务，必须是有 labels
        self.labels = labels
        # 是否是分类任务
        self.is_classification = is_classification
        # [(generated input and expected label, extracted content)]
        self.extraction_examples = extraction_examples
        assert task_instruction is not None and task_instruction != ""
        assert task_name is not None and task_name != ""
        assert metric in ["exact_match", "rouge"]
        assert examples != ""
        assert isinstance(extraction_examples, list) and len(extraction_examples) >= 2
        assert isinstance(is_classification, bool)
        assert ((labels != []) == is_classification)


task738 = Task(
    task_instruction="In this task you will be given a claim and a perspective. You should determine whether that perspective supports or undermines the claim. If the perspective could possibly convince someone with different view, it is supporting, otherwise it is undermining.",
    task_name="task738",
    examples="""
[input]="claim: Music containing lyrics that glorify violent and criminal lifestyles should be banned.
perspective: hip hop artists have a right to free speech"
[output]="undermine"

[input]="claim: Abolish the US Electoral College.
perspective: The electoral college weakens incentives for voting and party building."
[output]="support"
""".strip(),
expected_content="\"claim\" and \"perspective\"",
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore"],
metric="exact_match",
labels=["support", "undermine"],
extraction_examples=[
    (
        """"[input] = "claim: Music containing lyrics that glorify violent and criminal lifestyles should be banned. perspective: hip hop artists have a right to free speech." [expected output] = undermine.""",
        """claim: Music containing lyrics that glorify violent and criminal lifestyles should be banned. perspective: hip hop artists have a right to free speech."""
     ),
    ("""[input] = "claim: Abolish the US Electoral College. perspective: The Electoral College enhances nationwide voter participation and party building, ensuring that smaller states and rural areas have a greater voice in elections." [expected output] = support.""",
     """claim: Abolish the US Electoral College. perspective: The electoral college weakens incentives for voting and party building.""")
    ],
is_classification=True,
)

task1554 = Task(
    task_instruction="In this task, you are given two statements. The task is to output whether a given textual premise, i.e. Statement 2, entails or implies a given scientific fact, i.e. Statement 1. The output should be 'entails' if Statement 2 supports Statement 1 and should be 'neutral' otherwise.",
    task_name="task1554",
    examples="""
[input]="claim: Music containing lyrics that glorify violent and criminal lifestyles should be banned.
perspective: hip hop artists have a right to free speech"
[output]="undermine"

[input]="claim: Abolish the US Electoral College.
perspective: The electoral college weakens incentives for voting and party building."
[output]="support"
""".strip(),
expected_content="\"Sentence 1\" and \"Sentence 2\"",
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore"],
metric="exact_match",
labels=["entails", "neutral"],
extraction_examples=[
    (
        """"[input] = "claim: Music containing lyrics that glorify violent and criminal lifestyles should be banned. perspective: hip hop artists have a right to free speech." [expected output] = neutral""",
        """claim: Music containing lyrics that glorify violent and criminal lifestyles should be banned. perspective: hip hop artists have a right to free speech."""
     ),
    ("""[input] = "claim: Abolish the US Electoral College. perspective: The Electoral College enhances nationwide voter participation and party building, ensuring that smaller states and rural areas have a greater voice in elections." [expected output] = entails""",
     """claim: Abolish the US Electoral College. perspective: The electoral college weakens incentives for voting and party building.""")
    ],
is_classification=True,
)

task935 = Task(
    task_instruction="In this task, you are given a premise, a hypothesis, and an update. The premise sentence describes a real-world situation and is always assumed to be true. The hypothesis sentence describes an assumption or inference that you might make about that situation having read the premise. The update provides additional information about the situation that might weaken or strengthen the hypothesis. A weakener is a statement that weakens the hypothesis. It makes you much less likely to believe the hypothesis is true. A strengthener is a statement that strengthens the hypothesis. It makes you much more likely to believe the hypothesis is true. Your task is to output 'strengthener' or 'weakener' if the update strengths or weakens the hypothesis, respectively.",
    task_name="task935",
    examples="""
[input]="Premise: PersonX seems interested
Hypothesis: PersonX then good activity
Update: PersonX was a good student"
[output]="strengthener"

[input]="Premise: PersonX seems interested
Hypothesis: PersonX then good activity
Update: PersonX was faking to get close to a girl"
[output]="weakener"
""".strip(),
expected_content="\"Premise\", \"Hypothesis\", and \"Update\"",
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore"],
metric="exact_match",
labels=["strengthener", "weakener"],
extraction_examples=[
    (
        """"[input] = "Yeah. I can give a generated input as Premise: PersonX seems interested\nHypothesis: PersonX then good activity\nUpdate: PersonX was a good student" [expected output] = strengthener""",
        """Premise: PersonX seems interested\nHypothesis: PersonX then good activity\nUpdate: PersonX was a good student"""
     ),
    ("""[input] = "Premise: PersonX keeps my eyes open Hypothesis: Because PersonX wanted to examine my eyes Update: They are trying " [expected output] = weakener""",
     """claim: Premise: PersonX keeps my eyes open Hypothesis: Because PersonX wanted to examine my eyes Update: They are trying to talk to them and they want to sleep""")
    ],
is_classification=True,
)