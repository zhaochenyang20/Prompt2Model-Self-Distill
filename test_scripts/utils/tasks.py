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
[input]="Sentence 1: The sum of all chemical reactions that take place within an organism is known as metabolism. Sentence 2: Metabolism is the sum total of all chemical reactions performed by an organism."
[output]="entails"

[input]="Sentence 1: The endocrine system produces most of the hormones that regulate body functions. Sentence 2: Your endocrine glands produce hormones that control all your body functions."
[output]="entails"

[input]="Sentence 1: Warm and humid temperature and moisture conditions describe an air mass that originates over the Atlantic ocean near the equator. Sentence 2: Maritime tropical air Warm, humid air mass that forms over tropical and subtropical oceans."
[output]="neutral"
""".strip(),
expected_content="\"Sentence 1\" and \"Sentence 2\"",
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore"],
metric="exact_match",
labels=["entails", "neutral"],
extraction_examples=[
    (
        """"[input] = "Yes. I can generate a new intput. Sentence 1: The sum of all chemical reactions that take place within an organism is known as metabolism. Sentence 2: Metabolism is the sum total of all chemical reactions performed by an organism." [expected output] = entails""",
        """claim: Sentence 1: The sum of all chemical reactions that take place within an organism is known as metabolism. Sentence 2: Metabolism is the sum total of all chemical reactions performed by an organism."""
     ),
    ("""[input] = "Sentence 1: Warm and humid temperature and moisture conditions describe an air mass that originates over the Atlantic ocean near the equator. Sentence 2: Maritime tropical air " [expected output] = neutral""",
     """Sentence 1: Warm and humid temperature and moisture conditions describe an air mass that originates over the Atlantic ocean near the equator. Sentence 2: Maritime tropical air Warm, humid air mass that forms over tropical and subtropical oceans.""")
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
        """[input] = "Premise: PersonX seems interested\nHypothesis: PersonX then good activity\nUpdate: PersonX was a good student" [expected output] = strengthener""",
        """Premise: PersonX seems interested\nHypothesis: PersonX then good activity\nUpdate: PersonX was a good student"""
     ),
    (
        """[input] = "Premise: PersonX keeps my eyes open\nHypothesis: Because PersonX wanted to examine my eyes\nUpdate: They are trying" [expected output] = weakener""",
        """Premise: PersonX keeps my eyes open\nHypothesis: Because PersonX wanted to examine my eyes\nUpdate: They are trying"""
    )
    ],
is_classification=True,
)

task199 = Task(
    task_instruction="In this task, you're given a pair of sentences, sentence 1 and sentence 2. Your job is to determine if the two sentences clearly agree/disagree with each other, or if this can't be determined. Indicate your answer as yes or no respectively.",
    task_name="task199",
    examples="""
[input]="Sentence 1: Next to the MGM Grand you will find M and M World. Sentence 2: The candy has many fans who love its attractions."
[output]="no"

[input]="Sentence 1: I've forgotten his name now, confessed Tuppence. Sentence 2: Tuppence remembered his name later."
[output]="no"

[input]="Sentence 1: One of the first organizational realignments taking place is in the Office of the Taxpayer Advocate. Sentence 2: The office of the taxpayer advocate is having an organizational realignment."
[output]="yes"

[input]="Sentence 1: yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. Sentence 2: The tennis shoes have only one price."
[output]="yes"
""".strip(),
expected_content="\"Sentence 1\" and \"Sentence 2\"",
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore"],
metric="exact_match",
labels=["no", "yes"],
extraction_examples=[
    (
        """
        [input]="Sentence 1: Next to the MGM Grand you will find M and M World. Sentence 2: The candy has many " [expected output] = no
        """.strip(),
        """
        Sentence 1: Next to the MGM Grand you will find M and M World. Sentence 2: The candy has many fans who love its attractions.
        """.strip()
     ),
    (
        """
        [input] = "Yes. I can generate a new exmaple. Sentence 1: I've forgotten his name now, confessed Tuppence. Sentence 2: Tuppence remembered his name later."  [expected output] = no
        """.strip(),
        """
        Sentence 1: I've forgotten his name now, confessed Tuppence. Sentence 2: Tuppence remembered his name later.
        """.strip()
    ),
    (
        """
        [input]="Sentence 1: One of the first organizational realignments taking place is in the Office of the Taxpayer Advocate. Sentence 2: The office of the taxpayer advocate is having an organizational realignment." [expected output] = yes
        """.strip(),
        """
        Sentence 1: One of the first organizational realignments taking place is in the Office of the Taxpayer Advocate. Sentence 2: The office of the taxpayer advocate is having an organizational realignment.
        """.strip()
    ),
    (
        """
        [input]="Sentence 1: yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. Sentence 2: The tennis shoes have only one price." [expected output] = yes
        """.strip(),
        """
        Sentence 1: yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. Sentence 2: The tennis shoes have only one price.
        """.strip()
    )
    ],
is_classification=True,
)

task202 = Task(
    task_instruction="In this task, you're given a statement, and three sentences as choices. Your job is to determine which sentence clearly disagrees with the statement. Indicate your answer as '1', '2', or '3' corresponding to the choice number of the selected sentence.",
    task_name="task202",
    examples="""
        [input]="Statement: Next to the MGM Grand you will find M and M World, four stories of merchandise and memorabilia dedicated to the candy that doesn't melt in your hand. Choices: 1. The candy has many fans who love its attractions. 2. There's four stories of memorabilia dedicated to a candy. 3. That particular candy melts and becomes difficult to eat."
        [output]="3"
        
        [input]="Statement: I've forgotten his name now, confessed Tuppence. Choices: 1. Tuppence forgot his name. 2. Tuppence never could forget his name. 3. Tuppence remembered his name later."
        [output]="2"
        
        [input]="Statement: One of the first organizational realignments taking place is in the Office of the Taxpayer Advocate. Choices: 1. The office of the taxpayer advocate is the last to be realigned. 2. The realignment is taking place over a few weeks. 3. The office of the taxpayer advocate is having an organizational realignment."
        [output]="1"
        
        [input]="Statement: yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. Choices: 1. The tennis shoes have a range of prices. 2. The tennis shoes can be in the hundred dollar range. 3. The tennis shoes are not over hundred dollars."
        [output]="3"
""".strip(),
expected_content="""
    "Statement" and "Choices"
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore"],
metric="exact_match",
labels=["1", "2", "3"],
extraction_examples=[
    (
        """
[input] = "Statement: Next to the MGM Grand you will find M and M World, four stories of merchandise and memorabilia dedicated to the candy that doesn't melt in your hand.

Choices:

The candy has many fans who love its attractions.
There's four stories of memorabilia dedicated to a candy.
That particular candy melts and becomes difficult to eat.
The statement suggests that M&M's World is a place where you can find merchandise and memorabilia dedicated to the candy that doesn't melt in your hand. Choice 1 does not necessarily disagree with the statement, but it is not relevant to the main point. Choice 2 is related to the statement, but it does not necessarily disagree with it. Therefore, the answer is not 1 or 2." [expected output] = 3
        """.strip(),
        """
        Statement: Next to the MGM Grand you will find M and M World, four stories of merchandise and memorabilia dedicated to the candy that doesn't melt in your hand. Choices: 1. The candy has many fans who love its attractions. 2. There's four stories of memorabilia dedicated to a candy. 3. That particular candy melts and becomes difficult to eat.
        """.strip()
     ),
    (
        """
        [input] = "Improved Statement: Tuppence confessed that she forgot the name of a person.

Improved Choices:

1.Tuppence forgot his name.
2.Tuppence remembered his name later.
3.Tuppence never could forget his name.
The improved statement clearly states the main point of the original statement and the choices are relevant to the topic."  [expected output] = 3
        """.strip(),
        """
        Statement: Tuppence confessed that she forgot the name of a person. Choices: 1. Tuppence forgot his name. 2.Tuppence remembered his name later. 3. Tuppence never could forget his name. 
        """.strip()
    )
    ],
is_classification=True,
)