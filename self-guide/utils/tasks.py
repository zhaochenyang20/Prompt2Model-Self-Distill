# We config the tasks for classification in this files


class Task:
    def __init__(
        self,
        task_instruction,
        task_name,
        examples,
        optional_list,
        metric,
        labels,
        is_classification,
    ) -> None:
        self.task_instruction = task_instruction
        self.task_name = task_name
        self.examples = examples
        # add `optional_list` to Noise Filter
        self.optional_list = optional_list
        self.metric = metric
        self.labels = labels
        self.is_classification = is_classification
        assert task_instruction is not None and task_instruction != ""
        assert task_name is not None and task_name != ""
        assert metric in ["exact_match", "rouge"]
        assert examples != ""
        assert isinstance(is_classification, bool)
        assert (labels != []) == is_classification


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
    optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore"],
    metric="exact_match",
    labels=["support", "undermine"],
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
    optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore"],
    metric="exact_match",
    labels=["entails", "neutral"],
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
    optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore"],
    metric="exact_match",
    labels=["strengthener", "weakener"],
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
    optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore"],
    metric="exact_match",
    labels=["no", "yes"],
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
    optional_list=["input", "output", "\n\n", "\\_\\_", "therefore", "Therefore"],
    metric="exact_match",
    labels=["1", "2", "3"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["1", "0"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["Entailment", "Contradiction", "Neutral"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["1", "2", "3"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["Yes.", "No."],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["Contradiction", "Neutral", "Entailment"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["Contradiction", "Neutral", "Entailment"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
        "The premise",
        "The hypothesis",
    ],
    metric="exact_match",
    labels=["entails", "neutral"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["E", "C", "N"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["1", "2", "3"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["strengthener", "weakener"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["yes", "no"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["yes", "no"],
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["0", "1", "2"],
    is_classification=True,
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
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["neutral", "positive", "negated"],
    is_classification=True,
)

task1615 = Task(
    task_instruction="""In this task, given 2 input sentences, you must classify the relation between them. If the second sentence has a similar meaning to that of the first sentence then the output is 'B_entails_A', if the second sentence has the opposite meaning to the first sentence then it is classified as 'B_contradicts_A'. If you cannot clearly ascertain agreement/disagreement between the two sentences, the label is 'B_neutral_A'.""".strip(),
    task_name="task1615",
    examples="""
[input]="sentence_A: man is wearing a hard hat and dancing. sentence_B: There is no man with a hard hat dancing."
[output]="B_contradicts_A"

[input]="sentence_A: A baby is crying. sentence_B: A man is exercising."
[output]="B_neutral_A"

[input]="sentence_A: A tiger is pacing around a cage. sentence_B: A tiger is walking around a cage."
[output]="B_entails_A"
""".strip(),
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["B_entails_A", "B_neutral_A", "B_contradicts_A"],
    is_classification=True,
)

task329 = Task(
    task_instruction="""In this task, you will be presented with a text, a pronoun from the text, and two candidate names. You should determine what the pronoun refers to and classify the answers into A, B, or Neither. A and B here are referring to option A and option B. Position of the pronoun in the text is showed within two "_"s.""".strip(),
    task_name="task329",
    examples="""
[input]="He grew up in Evanston, Illinois the second oldest of five children including his brothers, Fred and Gordon and sisters, Marge (Peppy) and Marilyn. His high school days were spent at New Trier High School in Winnetka, Illinois. MacKenzie studied with Bernard Leach from 1949 to 1952. _His_ simple, wheel-thrown functional pottery is heavily influenced by the oriental aesthetic of Shoji Hamada and Kanjiro Kawai. , Pronoun: His , A: MacKenzie , B: Bernard Leach"
[output]="A"

[input]="Reb Chaim Yaakov's wife is the sister of Rabbi Moishe Sternbuch, as is the wife of Rabbi Meshulam Dovid Soloveitchik, making the two Rabbis his uncles. Reb Asher's brother Rabbi Shlomo Arieli is the author of a critical edition of the novallae of Rabbi Akiva Eiger. Before _his_ marriage, Rabbi Arieli studied in the Ponevezh Yeshiva headed by Rabbi Shmuel Rozovsky, and he later studied under his father-in-law in the Mirrer Yeshiva. , Pronoun: his , A: Reb Asher , B: Akiva Eiger"
[output]="Neither"

[input]="Kathleen Nott was born in Camberwell, London. Her father, Philip, was a lithographic printer, and her mother, Ellen, ran a boarding house in Brixton; Kathleen was their third daughter. _She_ was educated at Mary Datchelor Girls' School (now closed), London, before attending King's College, London. , Pronoun: She , A: Ellen , B: Kathleen"
[output]="B"
""".strip(),
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["A", "B", "Neither"],
    is_classification=True,
)

task284 = Task(
    task_instruction="""In this task, you are given a review of movie. Your task is to classify given movie review into two categories: 1) positive, and 2) negative based on its content.""".strip(),
    task_name="task284",
    examples="""
[input]="For a movie that gets no respect there sure are a lot of memorable quotes listed for this gem. Imagine a movie where Joe Piscopo is actually funny! Maureen Stapleton is a scene stealer. The Moroni character is an absolute scream. Watch for Alan The Skipper Hale jr. as a police Sgt."
[output]="positive"

[input]="Bizarre horror movie filled with famous faces but stolen by Cristina Raines (later of TV's Flamingo Road) as a pretty but somewhat unstable model with a gummy smile who is slated to pay for her attempted suicides by guarding the Gateway to Hell! The scenes with Raines modeling are very well captured, the mood music is perfect, Deborah Raffin is charming as Cristina's pal, but when Raines moves into a creepy Brooklyn Heights brownstone (inhabited by a blind priest on the top floor), things really start cooking. The neighbors, including a fantastically wicked Burgess Meredith and kinky couple Sylvia Miles & Beverly D'Angelo, are a diabolical lot, and Eli Wallach is great fun as a wily police detective. The movie is nearly a cross-pollination of Rosemary's Baby and The Exorcist--but what a combination! Based on the best-seller by Jeffrey Konvitz, The Sentinel is entertainingly spooky, full of shocks brought off well by director Michael Winner, who mounts a thoughtfully downbeat ending with skill."
[output]="positive"

[input]="I felt brain dead, I'll tell you. This is the worst film I have ever bought. (in my ignorance I thought this was the Peter Jackson film of the same name). The performances are so terrible they are laughable. The special effects have not stood the test of time and look dire. The script promotes that kind of TV movie, stare into the middle distance kind of acting. The cast look as if they have been taking lessons from Joey Tribbiani, they have one look each, and stick to it. Plus I have never been confused by a movie until I sat down to watch this. The is it a dream or no plot is so terrible that frustration sets in within a few minutes. Avoid like a plague."
[output]="negative"
""".strip(),
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["positive", "negative"],
    is_classification=True,
)

task346 = Task(
    task_instruction="""In this task, you will be presented with a question, a word, and a POS tag. You have to determine whether the part-of-speech tag of the given word in the question is equal to the given POS tag or not. Give your answer with True or False. Here is the Alphabetical list of part-of-speech tags used in this task: CC: Coordinating conjunction, CD: Cardinal number, DT: Determiner, EX: Existential there, FW: Foreign word, IN: Preposition or subordinating conjunction, JJ: Adjective, JJR: Adjective, comparative, JJS: Adjective, superlative, LS: List item marker, MD: Modal, NN: Noun, singular or mass, NNS: Noun, plural, NNP: Proper noun, singular, NNPS: Proper noun, plural, PDT: Predeterminer, POS: Possessive ending, PRP: Personal pronoun, PRP$: Possessive pronoun, RB: Adverb, RBR: Adverb, comparative, RBS: Adverb, superlative, RP: Particle, SYM: Symbol, TO: to, UH: Interjection, VB: Verb, base form, VBD: Verb, past tense, VBG: Verb, gerund or present participle, VBN: Verb, past participle, VBP: Verb, non-3rd person singular present, VBZ: Verb, 3rd person singular present, WDT: Wh-determiner, WP: Wh-pronoun, WP$: Possessive wh-pronoun, WRB: Wh-adverb""".strip(),
    task_name="task346",
    examples="""
[input]="Who were the builders of the mosque in Herat with fire temples ? , Word: Who , POS tag: IN"
[output]="False"

[input]="What is the borough in which Kia Oval is located ? , Word: is , POS tag: VBZ"
[output]="True"
""".strip(),
    optional_list=[
        "input",
        "output",
        "\n\n",
        "\\_\\_",
        "therefore",
        "Therefore",
        "Hence",
    ],
    metric="exact_match",
    labels=["False", "True"],
    is_classification=True,
)
