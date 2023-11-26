"""Test Input Generator."""

import os
from pathlib import Path

from prompt2model.output_annotator import VLLMPromptBasedOutputAnnotator
from prompt2model.prompt_parser import MockPromptSpec, TaskType

prompt_spec = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction="Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.",  # # noqa E501
    examples="""
[input]="Question: What city did Super Bowl 50 take place in? Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."
[output]="Santa Clara"

[input]="Question: What river runs through Warsaw? Context: Warsaw (Polish: Warszawa [varˈʂava] ( listen); see also other names) is the capital and largest city of Poland. It stands on the Vistula River in east-central Poland, roughly 260 kilometres (160 mi) from the Baltic Sea and 300 kilometres (190 mi) from the Carpathian Mountains. Its population is estimated at 1.740 million residents within a greater metropolitan area of 2.666 million residents, which makes Warsaw the 9th most-populous capital city in the European Union. The city limits cover 516.9 square kilometres (199.6 sq mi), while the metropolitan area covers 6,100.43 square kilometres (2,355.39 sq mi)."
[output]="Vistula River"

[input]="Question: The Ottoman empire controlled territory on three continents, Africa, Asia and which other? Context: The Ottoman Empire was an imperial state that lasted from 1299 to 1923. During the 16th and 17th centuries, in particular at the height of its power under the reign of Suleiman the Magnificent, the Ottoman Empire was a powerful multinational, multilingual empire controlling much of Southeast Europe, Western Asia, the Caucasus, North Africa, and the Horn of Africa. At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the empire, while others were granted various types of autonomy during the course of centuries."
[output]="Europe"
""",  # noqa E501
)

inputs = [
    '[context]="\n\n\n[instruction]\n\nYour task is to',
    "What city is famous for the Grand Canyon?",
    '\n[context]="\n\n"',
    "?[context]=?",
    "\n[inputifference]=1\n\nThe answer to the question",
    "\n[context]=",
    "\"Question: Which lake is the world's largest freshwater lake by surface area? Context: Lake Superior is the world's largest freshwater lake by surface area, located in North America. It is the largest of the Great Lakes and the second largest in the world, with a surface area of roughly 31,800 square miles and a deepest point of 1642 ft. It straddles the border",
    "\n?",
    "\n\\*Question: What is the capital of Poland?\n*Context: Warsaw is the capital and largest city of Poland. Located on the Vistula River, it is one of the oldest cities in Europe and has a rich history dating back over 1,000 years. In the 16th and 17th centuries, Warsaw was one of the largest and most powerful cities in Europe, and it played a prominent role in the Polish-Lithuanian Commonwealth. Today, Warsaw is a major cultural, scientific, and business center, with a population of over 1.7 million people.",
    "?",
    '"Question: In which year did the Wright brothers invent the first airplane?"\n\n(Note: This is a classic example of a question, and the context passage does not provide the answer directly.)',
    '\n"Which city is home to the Grand Canyon?"',
    "\n?[context]=Warsaw",
    '"Question: What river runs through',
    '"Question: What was the capital city of France in the 19th century?\nContext: During the 19th century, France was a major power in Europe, and the capital city saw significant growth and development. At the beginning of the century, the capital was Paris, which was also the largest city in the country. However, during the century, several other cities emerged as important centers of French culture, industry, and politics. The most prominent among these was Marseille, which became a major port and industrial center, while other cities such as Lyon, Bordeaux, and Strasbourg also developed significant economic and cultural importance during the 19th century."',
    '"Question: What was the capital of the Byzantine Empire during the reign of Justinian I? Context: The Byzantine Empire was a continuation of the Roman Empire, with its capital being Constantinople (now Istanbul, Turkey), establishing in 330 AD. Justinian I, who reigned from 527 to 565, is considered one of',
    '\n"Question: What was the capital of the Byzantine Empire? Context: The Byzantine Empire was a medieval empire that controlled a significant portion of the eastern Mediterranean and the western end of the Silk Road. It was ruled by the Byzantine dynasty, and its capital city was Constantinople (modern-day Istanbul, Turkey). From the early years of the 4th century to the fall of the Byzantine Empire in 1453, Constantinople was the largest and wealthiest city in the world, with a population of over a million people. The city was strategically located',
    "\nYou've generated a new [input]! Please provide the context after the colon (:), so I can help you with the [input].",
    "Question: What is the name of the largest city in the world?\n\nContext: As of 2021, the world's largest city by population is Tokyo, the capital and largest city of Japan. It is located in the Kantō region on the eastern side of the island of Honshu. With a population of over 37 million people, Tokyo is the most populous metropolitan area in the world. The city submerged much of its territory during the 2011 Tōhoku earthquake and tsunami, with damage to infrastructure and the loss of over 16,000 lives. The city is known for its high-tech industries, including robotics, gaming, and printing, as well as its iconic skyscrapers and modern art museums.\"",
    "\"Question: What is the tallest mountain in Africa? Context: The tallest mountain in Africa is Mount Kilimandscharo, with a height of 5,895 meters (19,341 ft) above sea level, located in the northern slice of Tanzania, East Africa. It's also a dormant volcano and the highest peak of the Kilimanjaro mountain group. The mountain stands out as the highest point in Africa and the highest freestanding mountain in the world, excluding Western mountains. Kilimandscharo is a popular tourist destination and has several routes for hiking. The mountain's name is often misspelled Kilimandjaro or Kili-Mandjarano and some other way.\"",
    "[input]=",
    '\n"What was the name of the city that hosted the first ever Olympic games?"',
]

output_annotator = VLLMPromptBasedOutputAnnotator()

generated_inputs_dir = Path("/home/cyzhao/generated_datasets")

output_dataset = output_annotator.annotate_outputs(
    input_strings=inputs,
    prompt_spec=prompt_spec,
    hyperparameter_choices={},
)
