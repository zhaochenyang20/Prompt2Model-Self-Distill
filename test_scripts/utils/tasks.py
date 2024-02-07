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
        # assert isinstance(extraction_examples, list) and len(extraction_examples) >= 2
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

task1344 = Task(
    task_instruction="In this task, you're given two sentences. Indicate if the first sentence clearly entails the second sentence (i.e., one can conclude the 2nd sentence by reading the 1st one). Indicate your answer with '1' if the first sentence entails the second sentence, otherwise answer with '0'.",
    task_name="task1344",
    examples="""   
        [input]="Sentence 1: No Weapons of Mass Destruction Found in Iraq Yet. Sentence 2: Weapons of Mass Destruction Found in Iraq."
        [output]="0"
        [input]="Sentence 1: A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI. Sentence 2: Pope Benedict XVI is the new leader of the Roman Catholic Church."
        [output]="1"
        [input]="Sentence 1: Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients. Sentence 2: Herceptin can be used to treat breast cancer."
        [output]="1"
        [input]="Sentence 1: Nearly 4 million children who have at least one parent who entered the U.S. illegally were born in the United States and are U.S. citizens as a result, according to the study conducted by the Pew Hispanic Center. That's about three quarters of the estimated 5.5 million children of illegal immigrants inside the United States, according to the study. About 1.8 million children of undocumented immigrants live in poverty, the study found. Sentence 2: Three quarters of U.S. illegal immigrants have children."
        [output]="0"
""".strip(),
expected_content="""
    "Sentence 1" and "Sentence 2"
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
metric="exact_match",
labels=["1", "0"],
extraction_examples=[
    (
        """
[input] = "Sentence 1: No Weapons of Mass Destruction Found in Iraq Yet.
Sentence 2: Weapons of Mass Destruction Found in Iraq.

The first sentence clearly states that no weapons of mass destruction have been found in Iraq yet." [expected output] = 0
        """.strip(),
        """
        Sentence 1: No Weapons of Mass Destruction Found in Iraq Yet. Sentence 2: Weapons of Mass Destruction Found in Iraq.
        """.strip()
     ),
    (
        """
        [input] = "Sentence 1: A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI. Sentence 2: Pope Benedict XVI is the new leader of the Roman Catholic Church." [expected output] = 0
        """.strip(),
        """
        Sentence 1: A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI. Sentence 2: Pope Benedict XVI is not the new leader of the Roman Catholic Church.
        """.strip()
    ),
    (
        """
        [input] = "Sentence 1: Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients. Sentence 2: Herceptin can be used to treat breast cancer.

Sentence 1 entails Sentence 2. Therefore, the answer is 1." [expected output] = 1
        """.strip(),
        """
        Sentence 1: Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients. Sentence 2: Herceptin can be used to treat breast cancer.
        """.strip()
    ),    
    ],
is_classification=True,
)

task1385 = Task(
    task_instruction="""In this task, you will be presented with a premise and a hypothesis sentence. Determine whether the hypothesis sentence entails (implies), contradicts (opposes), or is neutral with respect to the given premise sentence. Please answer with "Contradiction", "Neutral", or "Entailment".""",
    task_name="task1385",
    examples="""   
[input]="Premise: Lost Moon: The Perilous Voyage of Apollo 13 (published in paperback as Apollo 13), is a non-fiction book first published in 1994 by astronaut James Lovell and journalist Jeffrey Kluger, about the failed April 1970 Apollo 13 lunar landing mission which Lovell commanded. The book is the basis of the 1995 film adaptation "Apollo 13", directed by Ron Howard. <sep> Hypothesis: the book wouldnt have happened if we didnt try to go into space."
[output]="Entailment"
[input]="Premise: Linguistics is the scientific study of language, and involves an analysis of language form, language meaning, and language in context. The earliest activities in the documentation and description of language have been attributed to the 4th century BCE Indian grammarian Pāṇini, who wrote a formal description of the Sanskrit language in his "Aṣṭādhyāyī ". <sep> Hypothesis: Form and meaning are the only aspects of language linguistics is concerned with."
[output]="Contradiction"
[input]="Premise: Stephen Williams (born 5 June 1961) is a former Australian rules footballer in the South Australian National Football League, playing for the Port Adelaide Magpies and is currently an assistant development coach at Port Adelaide Power and head coach of the Immanuel College first XVIII. <sep> Hypothesis: Stephen Williams quit playing football due to an injury."
[output]="Neutral"
""".strip(),
expected_content="""
    "Premise" and "Hypothesis"
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
metric="exact_match",
labels=["Entailment", "Contradiction", "Neutral"],
extraction_examples=[
    (
        """
[input] = "Premise: Lost Moon: The Perilous Voyage of Apollo 13, a book about the failed Apollo 13 lunar landing mission, was written by astronaut James Lovell and journalist Jeffrey Kluger. It was published in 1994 and is the basis of the 1995 film "Apollo 13". <sep> Hypothesis: The book wouldn't have been written if we hadn't tried to go into space. Entailment" [expected output] = "Entailment"
        """.strip(),
        """
        Premise: Lost Moon: The Perilous Voyage of Apollo 13, a book about the failed Apollo 13 lunar landing mission, was written by astronaut James Lovell and journalist Jeffrey Kluger. It was published in 1994 and is the basis of the 1995 film "Apollo 13". <sep> Hypothesis: The book wouldn't have been written if we hadn't tried to go into space.
        """.strip()
     ),
    (
        """
        [input] = "Premise: Linguistics is the scientific study of language, and involves an analysis of language form, language meaning, and language in context. Hypothesis: Form and meaning are not the only aspects of language linguistics is concerned with." [expected output] = "Contradiction"
        """.strip(),
        """
        Premise: Linguistics is the scientific study of language, and involves an analysis of language form, language meaning, and language in context. <sep> Hypothesis: Form and meaning are the only aspects of language linguistics is concerned with.
        """.strip()
    )   
    ],
is_classification=True,
)

task201 = Task(
    task_instruction="""In this task, you're given a statement and three sentences as choices. Your job is to determine the neutral choice based on your inference from the statement and your commonsense knowledge. The neutral choice is a sentence that neither agrees nor disagrees with the statement. Indicate your answer as '1', '2', or '3', corresponding to the choice number of the selected sentence. If sentence X agrees with sentence Y, one's correctness follows from the other one. If sentence X disagrees with sentence Y, they can not be correct at the same time.""",
    task_name="task201",
    examples="""   
[input]="Statement: Next to the MGM Grand you will find M and M World, four stories of merchandise and memorabilia dedicated to the candy that doesn't melt in your hand. Choices: 1. The candy has many fans who love its attractions. 2. There's four stories of memorabilia dedicated to a candy. 3. That particular candy melts and becomes difficult to eat."
[output]="1"
[input]="Statment: I've forgotten his name now, confessed Tuppence. Choices: 1. Tuppence forgot his name. 2.Tuppence remembered his name later. 3. Tuppence never could forget his name."
[output]="2"
[input]="Statement: One of the first organizational realignments taking place is in the Office of the Taxpayer Advocate. Choices: 1. The office of the taxpayer advocate is the last to be realigned. 2. The office of the taxpayer advocate is having an organizational realignment. 3. The realignment is taking place over a few weeks."
[output]="3"
[input]="Statement: yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. Choices: 1. The tennis shoes have a range of prices. 2. The tennis shoes can be in the hundred dollar range. 3. The tennis shoes are not over hundred dollars."
[output]="1"
""".strip(),
expected_content="""
    "Statement" and "Choices"
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
metric="exact_match",
labels=["1", "2", "3"],
extraction_examples=[
    (
        """
        [input] = "Statement: The company has a new policy that requires all employees to work from home at least two days a week. Choices: 1. The company is not allowing employees to work from home. 2. The company has implemented a new policy. 3. The company is not requiring employees to work from home. The statement suggests that the company has a new policy that requires all employees to work from home at least two days a week. Choice 1 does not necessarily disagree with the statement, but it is not relevant to the main point." [expected output] = 2
        """.strip(),
        """
        Statement: The company has a new policy that requires all employees to work from home at least two days a week. Choices: 1. The company is not allowing employees to work from home. 2. The company has implemented a new policy. 3. The company is not requiring employees to work from home.
        """.strip()
     ),
    (
        """
        [input] = "Statement: I\'ve heard that some people are considering a new type of renewable energy source that involves harnessing the power of ocean currents. Choices: 1. The ocean currents can be harnessed for renewable energy. 2. The new type of renewable energy source is not practical. 3. The new type of renewable energy source is not cost-effective." [expected output] = 1
        """.strip(),
        """
        Statement: I\'ve heard that some people are considering a new type of renewable energy source that involves harnessing the power of ocean currents. Choices: 1. The ocean currents can be harnessed for renewable energy. 2. The new type of renewable energy source is not practical. 3. The new type of renewable energy source is not cost-effective.
        """.strip()
    )   
    ],
is_classification=True,
)

task020 = Task(
    task_instruction="""The answer will be 'yes' if the provided sentence contains an explicit mention that answers the given question. Otherwise, the answer should be 'no'. Instances where the answer is implied from the sentence using "instinct" or "common sense" (as opposed to being written explicitly in the sentence) should be labeled as 'no'.""",
    task_name="task020",
    examples="""   
[input]="Sentence: Jack played basketball for an hour after school, after which he was very tired. \nQuestion: How long did Jack play basketball?"
[output]="Yes."
[input]="Sentence: He was born in China, so he went to the Embassy at 1 pm to apply for a U.S. Visa. \nQuestion: When did he go to Embassy?"
[output]="Yes."
""".strip(),
expected_content="""
    "Sentence" and "Question"
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
metric="exact_match",
labels=["Yes.", "No."],
extraction_examples=[
    (
        """
        [input] = "Yes. There is a new input for this tasks. Sentence: She was born in Canada and moved to the United States when she was 20 years old. \nQuestion: When did she move to the United States?" [expected output] = Yes.
        """.strip(),
        """
        Sentence: She was born in Canada and moved to the United States when she was 20 years old. \nQuestion: When did she move to the United States?
        """.strip()
     ),
    (
        """
        [input] = "Sentence: She went to the store to buy bread, milk, and eggs.\nQuestion: When did she go to the store?" [expected output] = No.
        """.strip(),
        """
        Sentence: She went to the store to buy bread, milk, and eggs.\nQuestion: When did she go to the store?
        """.strip()
    )   
    ],
is_classification=True,
)

task1386 = Task(
    task_instruction="""In this task, you will be presented with a premise and a hypothesis sentence. Determine whether the hypothesis sentence entails (implies), contradicts (opposes), or is neutral with respect to the given premise. Please answer with "Contradiction", "Neutral", or "Entailment".""",
    task_name="task1386",
    examples="""   
[input]="Premise: The Washington Nationals are a professional baseball team that has been based in Washington, D.C. since . The Nationals are a member of both the Major League Baseball's (MLB) National League Eastern Division and the National League (NL) itself. Since the 2008 season, the Nationals have played in Nationals Park; from 2005 through , the team played in Robert F. Kennedy Memorial Stadium. <sep> Hypothesis: The Washington Nationals have played in Nationals Park for more than 1000 days."
[output]="Entailment"
[input]="Premise: The 1986–87 St. John's Redmen basketball team represented St. John's University during the 1986–87 NCAA Division I men's basketball season. The team was coached by Lou Carnesecca in his nineteenth year at the school. St. John's home games are played at Alumni Hall and Madison Square Garden and the team is a member of the Big East Conference. <sep> Hypothesis: Lou Carnesecca's first season as coach of the St. John's Redmen basketball team was in 1974."
[output]="Contradiction"
[input]="Premise: Clear Hearts Grey Flowers is the second full-length and final album by Jack Off Jill. Produced by Chris Vrenna of Nine Inch Nails/Tweaker, it was released in July 2000 on the now-defunct label Risk Records. After "Clear Hearts, Grey Flowers" the band formally split up and moved on to establish other projects. <sep> Hypothesis: Risk Records released Jack Off Jill's initial album."
[output]="Neutral"
""".strip(),
expected_content="""
    "Premise" and "Hypothesis"
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
metric="exact_match",
labels=["Contradiction", "Neutral", "Entailment"],
extraction_examples=[
    (
        """
        [input] = "Premise: The 2022 FIFA World Cup will be the 22nd edition of the quadrennial international men's soccer tournament, and is scheduled to be held in Qatar from 21 November to 18 December 2022. The tournament will be hosted by the Qatar Football Association and will be the first time that the World Cup will be held in the Middle East. The tournament will be played across eight venues in five cities: Al Rayyan, Doha, Lusail, Al Khor, and Ras Abu Aboud. <sep> Hypothesis: The 2022 FIFA World Cup"  [expected output] = Entailment
        """.strip(),
        """
        Premise: The 2022 FIFA World Cup will be the 22nd edition of the quadrennial international men's soccer tournament, and is scheduled to be held in Qatar from 21 November to 18 December 2022. The tournament will be hosted by the Qatar Football Association and will be the first time that the World Cup will be held in the Middle East. The tournament will be played across eight venues in five cities: Al Rayyan, Doha, Lusail, Al Khor, and Ras Abu Aboud. <sep> Hypothesis: The 2022 FIFA World Cup will be held in Qatar.
        """.strip()
     ),
    (
        """
        [input] = "Premise: The New York Yankees are a professional baseball team based in New York City. They are members of Major League Baseball\'s (MLB) American League. The Yankees have won the most World Series titles with 27 victories. They also have the highest win percentage of any team in Major League Baseball history with a winning percentage of .698. The Yankees have played their home games at Yankee Stadium since 1923. <sep> Hypothesis: The New York Yankees have played at least one game at Wrigley Field." [expected content] = Contradiction
        """.strip(),
        """
        Premise: The New York Yankees are a professional baseball team based in New York City. They are members of Major League Baseball\'s (MLB) American League. The Yankees have won the most World Series titles with 27 victories. They also have the highest win percentage of any team in Major League Baseball history with a winning percentage of .698. The Yankees have played their home games at Yankee Stadium since 1923. <sep> Hypothesis: The New York Yankees have played at least one game at Wrigley Field.
        """.strip()
    )   
    ],
is_classification=True,
)

task1388 = Task(
    task_instruction="""In this task, you will be presented with a premise and a hypothesis sentence. Determine whether the hypothesis sentence entails (implies), contradicts (opposes), or is neutral with respect to the given premise. Please answer with "Contradiction", "Neutral", or "Entailment".""",
    task_name="task1388",
    examples="""   
[input]="Premise: Ockleton, Morpurgo, Cornelius, Dysart and half a dozen others too drunk to mention. But there was so much coming and going that any one of us could have slipped out, pushed Everett through the window and slipped back again without being noticed. Damn it all we didn't even notice Everett was missing until a porter tripped over him in the quad so anything's theoretically possible. <sep> Hypothesis: Everett was missing"
[output]="Entailment"
[input]="Premise: I should dearly have liked to know whether they were Europeans or Americans, but I couldn't hear the accents. They appeared to be arguing. I hoped the white men weren't telling him to eliminate all witnesses because I don't believe it would have needed much persuasion. <sep> Hypothesis: eliminating all witnesses would have needed much persuasion"
[output]="Contradiction"
[input]="Premise: B: All right, well. A: Um, short term, I don't think anything's going to be done about it or probably should be done about it. B: Right.  Uh, are you saying you don't think anything should be done in the short term? <sep> Hypothesis: anything should be done in the short term"
[output]="Neutral"
""".strip(),
expected_content="""
    "Premise" and "Hypothesis"
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
metric="exact_match",
labels=["Contradiction", "Neutral", "Entailment"],
extraction_examples=[
    (
        """
        [input] = "Yes. There is a new input for this task. Premise: The company is planning to launch a new product soon. Hypothesis: The new product will be a huge success." [expected output] = Neutral
        """.strip(),
        """
        Premise: The company is planning to launch a new product soon. Hypothesis: The new product will be a huge success.
        """.strip()
     ),
    (
        """
        [input] = "Premise: The train is moving at a constant speed. <sep> Hypothesis: The train will stop in five minutes" [expected output] = Entailment
        """.strip(),
        """
        Premise: The train is moving at a constant speed. <sep> Hypothesis: The train is moving smoothly.
        """.strip()
    )  
    ],
is_classification=True,
)

task1529 = Task(
    task_instruction="""You are given two sentences. You have to find if there is entailment or agreement of the Hypothesis by the Premise. From the given pair of sentences, you should identify if there is enough information in the Premise to support the claim made in the Hypothesis. The Premise may not exactly be the same as Hypothesis. Your task is to return 'entails' if the premise supports hypothesis else return 'neutral'.""",
    task_name="task1529",
    examples="""   
[input]="Premise: Lyme Disease is caused by a bacterium that's transmitted by tick bite, but many infected people don't remember a bite. \n Hypothesis: Lyme disease is caused by bacteria."
[output]="entails"
[input]="Premise: Corolla Collective term for all the petals of a flower, these petals may be separate or fused together. \n Hypothesis: All of the petals together are called a corolla."
[output]="entails"
[input]="Premise: This can be dangerous to both plants and animals. \n Hypothesis: Nematodes can be a parasite of both."
[output]="neutral"
[input]="Premise: The liver is divided into the right lobe and left lobes. \n Hypothesis: The gallbladder is near the right lobe of the liver."
[output]="neutral"
""".strip(),
expected_content="""
    "Premise" and "Hypothesis"
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence", "The premise", "The hypothesis"],
metric="exact_match",
labels=["entails", "neutral"],
extraction_examples=[
    (
        """
        [input]="Yes. There is a new input for this task. Premise: Lyme Disease is caused by a bacterium that's transmitted by tick bite, but many infected people don't remember a bite. \n Hypothesis: Lyme disease is caused by bacteria." [expected output] = entails
        """.strip(),
        """
        Premise: Lyme Disease is caused by a bacterium that's transmitted by tick bite, but many infected people don't remember a bite. \n Hypothesis: Lyme disease is caused by bacteria.
        """.strip()
     ),
    (
        """
        [input]="Premise: The liver is divided into the right lobe and left lobes. \n Hypothesis: The right lobe and left lobe may exhibit differences in the generation of certain metabolic pathways or metabolic products." [expected output] = neutral
        """.strip(),
        """
        Premise: The liver is divided into the right lobe and left lobes. \n Hypothesis: The gallbladder is near the right lobe of the liver.
        """.strip()
    )  
    ],
is_classification=True,
)

task190 = Task(
    task_instruction="""In this task, you're given a pair of sentences, sentence 1 and sentence 2. Your job is to choose whether the two sentences clearly agree (entailment)/disagree (contradiction) with each other, or if this cannot be determined (neutral). Your answer must be in the form of the letters E, C, and N respectively.""",
    task_name="task190",
    examples="""   
[input]="Sentence 1: Jon saw his friend Tom coming out of the grocery store with a bag of fruit. Sentence 2: Tom had been shopping for fruit to give Jon."
[output]="N"
[input]="Sentence 1: The girl transferred all the flowers from the boquet to a vase. Sentence 2: The flowers will soon wither."
[output]="N"
[input]="Sentence 1: The skier was on the edge of the ramp. Sentence 2: The skier was dressed in winter clothes."
[output]="E"
[input]="Sentence 1: Joyce likes to eat fruit salad as often as possible. Sentence 2: Joyce loves eating healthy."
[output]="E"
[input]="Sentence 1: The boy skated down the staircase railing. Sentence 2: The boy is a newbie skater."
[output]="C"
[input]="Sentence 1: Bertha was selected as captain of her basketball team. Sentence 2: Bertha was not atheletically inclined."
[output]="C"
""".strip(),
expected_content="""
    "Sentence 1" and "Sentence 2"
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
metric="exact_match",
labels=["E", "C", "N"],
extraction_examples=[
    (
        """
        [input]="Yes. There is a new input for this task. Sentence 1: Jon saw his friend Tom coming out of the grocery store with a bag of fruit. Sentence 2: Tom had been shopping for fruit to give Jon." [expected output] = N
        """.strip(),
        """
        Sentence 1: Jon saw his friend Tom coming out of the grocery store with a bag of fruit. Sentence 2: Tom had been shopping for fruit to give Jon.
        """.strip()
    ),
    (
        """
        [input]="Sentence 1: The girl transferred all the flowers from the boquet to a vase. Sentence 2: The flower can live longer." [expected output] = N
        """.strip(),
        """
        Sentence 1: The girl transferred all the flowers from the boquet to a vase. Sentence 2: The flowers will soon wither.
        """.strip()
    ),
    (
        """
        [input]="Yes. There is a new input for this task. Sentence 1: The skier was on the edge of the ramp. Sentence 2: The skier was dressed in winter clothes." [expected output] = E
        """.strip(),
        """
        Sentence 1: The skier was on the edge of the ramp. Sentence 2: The skier was dressed in winter clothes.
        """.strip()
    ),
    (
        """
        [input]="Sentence 1: Joyce likes to eat fruit salad as often as possible. Sentence 2: Joyce does not love eating healthy." [expected output] = E
        """.strip(),
        """
        Sentence 1: Joyce likes to eat fruit salad as often as possible. Sentence 2: Joyce loves eating healthy.
        """.strip()
    ),
    (
        """
        [input]="Yes. There is a new input for this task. Sentence 1: The boy skated down the staircase railing. Sentence 2: The boy is a newbie skater." [expected output] = C
        """.strip(),
        """
        Sentence 1: The boy skated down the staircase railing. Sentence 2: The boy is a newbie skater.
        """.strip()
    ),
    (
        """
        [input]="Sentence 1: Bertha was selected as captain of her basketball team. Sentence 2: Bertha was atheletically inclined." [expected output] = C
        """.strip(),
        """
        Sentence 1: Bertha was selected as captain of her basketball team. Sentence 2: Bertha was not atheletically inclined.
        """.strip()
    )
    ],
is_classification=True,
)

task200 = Task(
    task_instruction="""In this task, you're given a statement and three sentences as choices. Your job is to determine which sentence can be inferred from the statement. Incorrect choices change the meaning in important ways or have details that are not mentioned in the statement. Indicate your answer as 1,2, or 3 corresponding to the choice number of the selected sentence.""",
    task_name="task200",
    examples="""   
[input]="Statement: Next to the MGM Grand you will find M and M World, four stories of merchandise and memorabilia dedicated to the candy that doesn't melt in your hand. Choices: 1. The candy has many fans who love its attractions. 2. There's four stories of memorabilia dedicated to a candy. 3. That particular candy melts and becomes difficult to eat."
[output]="2"
[input]="Statment: I've forgotten his name now, confessed Tuppence. Choices: 1. Tuppence forgot his name. 2.Tuppence remembered his name later. 3. Tuppence never could forget his name."
[output]="1"
[input]="Statement: One of the first organizational realignments taking place is in the Office of the Taxpayer Advocate. Choices: 1. The office of the taxpayer advocate is the last to be realigned. 2. The realignment is taking place over a few weeks. 3. The office of the taxpayer advocate is having an organizational realignment."
[output]="3"
""".strip(),
expected_content="""
    "Statement" and "Choices"
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
metric="exact_match",
labels=["1", "2", "3"],
extraction_examples=[
    (
        """
        [input]="Yes. There is a new input for this task. Statement: Next to the MGM Grand you will find M and M World, four stories of merchandise and memorabilia dedicated to the candy that doesn't melt in your hand. Choices: 1. The candy has many fans who love its attractions. 2. There's four stories of memorabilia dedicated to a candy. 3. That particular candy melts and becomes difficult to eat." [expected output] = 2
        """.strip(),
        """
        Statement: Next to the MGM Grand you will find M and M World, four stories of merchandise and memorabilia dedicated to the candy that doesn't melt in your hand. Choices: 1. The candy has many fans who love its attractions. 2. There's four stories of memorabilia dedicated to a candy. 3. That particular candy melts and becomes difficult to eat.
        """.strip()
    ),
    (
        """
        [input]="Statement: yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. Choices: 1. The tennis shoes can be in the hundred dollar range. 2. The tennis shoes have a range of prices. 3. The tennis shoes are not over hundred dollars." [expected output] = 2
        """.strip(),
        """
        Statement: yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. Choices: 1. The tennis shoes have a range of prices. 2. The tennis shoes can be in the hundred dollar range. 3. The tennis shoes are not over hundred dollars.
        """.strip()
    ),
    (
        """
        [input]="Yes. There is a new input for this task. Statment: I've forgotten his name now, confessed Tuppence. Choices: 1. Tuppence forgot his name. 2.Tuppence remembered his name later. 3. Tuppence never could forget his name." [expected output] = 1
        """.strip(),
        """
        Statment: I've forgotten his name now, confessed Tuppence. Choices: 1. Tuppence forgot his name. 2.Tuppence remembered his name later. 3. Tuppence never could forget his name.
        """.strip()
    )
    ],
is_classification=True,
)

task937 = Task(
    task_instruction="""In this task, you are given a hypothesis and an update. The hypothesis sentence is a statement that speaks of a socially normative behavior. In other words, it is a generalizing statement about how we expect people to behave in society. The update provides additional contexts about the situation that might UNDERMINE or SUPPORT the generalization. An undermining context provides a situation that weakens the hypothesis. A supporting context provides a situation that strengthens the generalization. Your task is to output 'strengthener' or 'weakener' if the update supports or undermines the hypothesis, respectively""",
    task_name="task937",
    examples="""   
[input]="Hypothesis: You should help your family with funeral expenses.\nUpdate: They have asked you to chip in"
[output]="strengthener"
[input]="Hypothesis: It's good to protect your property.\nUpdate: you don't care what happens to your property."
[output]="weakener"
[input]="Hypothesis: You should help your family with funeral expenses.\nUpdate: You are not financially stable to help out"
[output]="weakener"
""".strip(),
expected_content="""
    "Hypothesis" and "Update"
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
metric="exact_match",
labels=["strengthener", "weakener"],
extraction_examples=[
    (
        """
        [input]="Yes. There is a new input for this task. Hypothesis: You should help your family with funeral expenses.\nUpdate: They have asked you to chip in" [expected output] = strengthener
        """.strip(),
        """
        Hypothesis: You should help your family with funeral expenses.\nUpdate: They have asked you to chip in
        """.strip()
    ),
    (
        """
        [input]="Yes. There is a new input for this task. Hypothesis: It's good to protect your property.\nUpdate: you don't care what happens to your property." [expected output] = weakener
        """.strip(),
        """
        Hypothesis: It's good to protect your property.\nUpdate: you don't care what happens to your property.
        """.strip()
    ),
    (
        """
        [input]="Yes. There is a new input for this task. Hypothesis: You should help your family with funeral expenses.\nUpdate: You are not financially stable to help out" [expected output] = weakener
        """.strip(),
        """
        Hypothesis: You should help your family with funeral expenses.\nUpdate: You are not financially stable to help out
        """.strip()
    )
    ],
is_classification=True,
)

task642 = Task(
    task_instruction="""Given Sentence 1 and Sentence 2, indicate your answer as yes when the two sentences clearly agree or clearly disagree with each other. If the relationship cannot be determined, answer with 'no'.""",
    task_name="task642",
    examples="""   
[input]="A boy in a blue jacket rides a skateboard down the street. <sep> A blue jacketed boy is standing in football field."
[output]="yes"
[input]="A large group of people gather at a piano bar. <sep> A group of men at a piano bar."
[output]="no"
""".strip(),
expected_content="""
    2 sentences
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
metric="exact_match",
labels=["yes", "no"],
extraction_examples=[
    (
        """
        [input]="A boy in a blue jacket rides a skateboard down the street. <sep> A blue jacketed boy wears a hat." [expected output] = yes
        """.strip(),
        """
        A boy in a blue jacket rides a skateboard down the street. <sep> A blue jacketed boy is standing in football field.
        """.strip()
    ),
    (
        """
        [input]="Sentece 1: A large group of people gather at a piano bar. Sentence 2:A group of men at a piano bar." [expected output] = no
        """.strip(),
        """
        A large group of people gather at a piano bar. <sep> A group of men at a piano bar.
        """.strip()
    )
    ],
is_classification=True,
)

task642 = Task(
    task_instruction="""Given Sentence 1 and Sentence 2, indicate your answer as yes when the two sentences clearly agree or clearly disagree with each other. If the relationship cannot be determined, answer with 'no'.""",
    task_name="task642",
    examples="""   
[input]="A boy in a blue jacket rides a skateboard down the street. <sep> A blue jacketed boy is standing in football field."
[output]="yes"
[input]="A large group of people gather at a piano bar. <sep> A group of men at a piano bar."
[output]="no"
""".strip(),
expected_content="""
    2 sentences
    """.strip(),
optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore", "Hence"],
metric="exact_match",
labels=["yes", "no"],
extraction_examples=[
    (
        """
        [input]="A boy in a blue jacket rides a skateboard down the street. <sep> A blue jacketed boy wears a hat." [expected output] = yes
        """.strip(),
        """
        A boy in a blue jacket rides a skateboard down the street. <sep> A blue jacketed boy is standing in football field.
        """.strip()
    ),
    (
        """
        [input]="Sentece 1: A large group of people gather at a piano bar. Sentence 2:A group of men at a piano bar." [expected output] = no
        """.strip(),
        """
        A large group of people gather at a piano bar. <sep> A group of men at a piano bar.
        """.strip()
    )
    ],
is_classification=True,
)


task1612 = Task(
    task_instruction="""In this task, you're given a pair of sentences, sentence 1 and sentence 2. Your job is to choose whether the two sentences clearly agree (entailment)/disagree (contradiction) with each other, or if this cannot be determined (neutral). Your answer must be in the form of the numbers 0 (entailment), 1 (neutral), or 2(contradiction).""".strip(),
    task_name="task1612",
    examples="""
[input]="sentence_A: A dancer is dancing on the stage. sentence_B: A girl is giving dance performance on the dais."
[output]="0"

[input]="sentence_A: The crowd is cheering at her dance performance. sentence_B: The group is enjoying while eating food."
[output]="1"

[input]="sentence_A: A man is standing and has tears of joy seeing the dance performance. sentence_B: There is no man standing with happiness seeing the dance."
[output]="2"

""".strip(),
    expected_content="""
    "sentence_A" and "sentence_B"
    """.strip(),
    optional_list=["input", "output", "\n\n", "\\_\\_", "\\_", "therefore", "Therefore", "Hence"],
    metric="exact_match",
    labels=['0', '1', '2'],
    is_classification=True,
    extraction_examples=[
    (
        """
        [input]="sentence_A: A dancer is dancing on the stage. sentence_B: A girl is exercising in the gym." [expected output] = 0
        """.strip(),
        """
        sentence_A: A dancer is dancing on the stage. sentence_B: A girl is giving dance performance on the dais.
        """.strip()
    ),
    (
        """
        [input]="sentence_A: The crowd is cheering at her dance performance. sentence_B: The group dislikes her performance." [expected output] = 1
        """.strip(),
        """
        sentence_A: The crowd is cheering at her dance performance. sentence_B: The group is enjoying while eating food.
        """.strip()
    ),
        (
        """
        [input]="sentence_A: A man is standing and has tears of joy seeing the dance performance. sentence_B: There is a man standing with happiness seeing the dance." [expected output] = 2
        """.strip(),
        """
        sentence_A: A man is standing and has tears of joy seeing the dance performance. sentence_B: There is no man standing with happiness seeing the dance.
        """.strip()
    )
    ]
)
            
task1516 = Task(
    task_instruction="""In this task, you are given a premise and hypothesis. The task is to classify them into three categories: 'positive' if the hypothesis supports the premise, 'negated' if it opposes the premise, and 'neutral' if it neither supports nor opposes it.""".strip(),
    task_name="task1516",
    examples="""
[input]="'Premise : All ten guys that proved to boast were divorcing.','Hypothesis : There are exactly ten guys that proved to boast.'"
[output]="positive"

[input]="'Premise : All ten reports that can bore some waiter aren't disagreeing with Naomi.','Hypothesis : There are exactly eleven reports that can bore some waiter.'"
[output]="negated"

[input]="Premise : All ten guys that proved to boast weren't divorcing.','Hypothesis : There are exactly ten senators that proved to boast.'"
[output]="neutral"

""".strip(),
    expected_content="""
    "Premise" and "Hypothesis"
    """.strip(),
    optional_list=['input', 'output', '\n\n', '\\_\\_', 'therefore', 'Therefore', 'Hence'],
    metric="exact_match",
    labels=['neutral', 'positive', 'negated'],
    is_classification=True,
    extraction_examples=[
    (
        """
        [input]="'Premise : All ten guys that proved to boast were divorcing.','Hypothesis : There are more than ten guys that proved to boast.'" [expected output] = "positive"
        """.strip(),
        """
        'Premise : All ten guys that proved to boast were divorcing.','Hypothesis : There are exactly ten guys that proved to boast.'
        """.strip()
    ),
    (
        """
        [input]="'Premise : All ten reports that can bore some waiter aren't disagreeing with Naomi.','Hypothesis : There are exactly ten reports that can bore some waiter.'" [expected output] = "negated"
        """.strip(),
        """
        'Premise : All ten reports that can bore some waiter aren't disagreeing with Naomi.','Hypothesis : There are exactly eleven reports that can bore some waiter.'
        """.strip()
    ),
        (
        """
        [input]="Premise : All ten guys that proved to boast weren't divorcing.','Hypothesis : There are exactly ten guys that proved to boast.'" [expected output] = "neutral"
        """.strip(),
        """
        Premise : All ten guys that proved to boast weren't divorcing.','Hypothesis : There are exactly ten senators that proved to boast.'
        """.strip()
    )
    ]
)
            
task1615 = Task(
    task_instruction="""In this task, given 2 input sentences, you must classify the relation between them. If the second sentence has a similar meaning to that of the first sentence then the output is 'B_entails_A', if the second sentence has the opposite meaning to the first sentence then it is classified as 'B_contradicts_A'. If you cannot clearly ascertain agreement/disagreement between the two sentences, the label is 'B_neutral_A'.""".strip(),
    task_name="task1615",
    examples="""
[input]="sentence_A: man is wearing a hard hat and dancing. sentence_B: There is no man with a hard hat dancing."
[output]="B_contradicts_A"

[input]="sentence_A: A baby is crying. sentence_B: A man is exercising."
[output]="B_neutral_A"

[input]="sentence_A: A tiger is pacing around a cage. sentence_B: A tiger is walking around a cage"
[output]="B_entails_A"
""".strip(),
    expected_content=""""sentence_A" and "sentence_B"
    """.strip(),
    optional_list=['input', 'output', '\n\n', '\\_\\_', 'therefore', 'Therefore', 'Hence'],
    metric="exact_match",
    labels=['B_entails_A', 'B_neutral_A', 'B_contradicts_A'],
    is_classification=True,
    extraction_examples=[
    (
        """
        [input]="'Premise : All ten guys that proved to boast were divorcing.','Hypothesis : There are more than ten guys that proved to boast.'" [expected output] = "positive"
        """.strip(),
        """
        'Premise : All ten guys that proved to boast were divorcing.','Hypothesis : There are exactly ten guys that proved to boast.'
        """.strip()
    ),
    (
        """
        [input]="'Premise : All ten reports that can bore some waiter aren't disagreeing with Naomi.','Hypothesis : There are exactly ten reports that can bore some waiter.'" [expected output] = "negated"
        """.strip(),
        """
        'Premise : All ten reports that can bore some waiter aren't disagreeing with Naomi.','Hypothesis : There are exactly eleven reports that can bore some waiter.'
        """.strip()
    ),
        (
        """
        [input]="Premise : All ten guys that proved to boast weren't divorcing.','Hypothesis : There are exactly ten guys that proved to boast.'" [expected output] = "neutral"
        """.strip(),
        """
        Premise : All ten guys that proved to boast weren't divorcing.','Hypothesis : There are exactly ten senators that proved to boast.'
        """.strip()
    )
    ]
)