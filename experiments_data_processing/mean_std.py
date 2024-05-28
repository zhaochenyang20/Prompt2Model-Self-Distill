import re
import numpy as np

examples = {
    "task1345" : "[input]=\"What can one do after MBBS?\"\n[output]=\"What do i do after my MBBS ?\"\n\n[input]=\"Which is the best book to study TENSOR for general relativity from basic?\"\n[output]=\"Which is the best book for tensor calculus?\"\n\n[input]=\"What are the coolest Android hacks and tricks you know?\"\n[output]=\"What are some cool hacks for Android phones?\"\n\n[input]=\"Which are the best motivational videos?\"\n[output]=\"What are some of the best motivational clips?\"\n\n",
    "task281"  : "[input]=\"1: Four employees of the store have been arrested , but its manager -- herself a woman -- was still at large Saturday , said Goa police superintendent Kartik Kashyap . 2: If convicted , they could spend up to three years in jail , Kashyap said . 3: The four store workers arrested could spend 3 years each in prison if convicted .\"\n[output]=\"1: Four employees of the store 2: they 3: The four store workers\"\n\n[input]=\"1: Stewart said that she and her husband , Joseph Naaman , booked Felix on their Etihad Airways flight from the United Arab Emirates to New York 's John F . Kennedy International Airport on April 1 . 2: The couple said they spent $ 1,200 to ship Felix on the 14-hour flight . 3: Couple spends $ 1,200 to ship their cat , Felix , on a flight from the United Arab Emirates .\"\n[output]=\"1: their Etihad Airways flight from the United Arab Emirates to New York 's John F . Kennedy International Airport 2: the 14-hour flight 3: a flight from the United Arab Emirates\"\n\n[input]=\"1: But an Arizona official told CNN Bates never trained with the agency . 2:  He did n't come to Arizona ,  the official from the Maricopa County Sheriff 's Office said ,  and he certainly did n't train with us .  3: Maricopa County Sheriff 's Office in Arizona says Robert Bates never trained with them .\"\n[output]=\"1: never trained 2: did n't train 3: never trained\"\n\n",
    "task1562" : "[input]=\"Does this dog breed have short legs compared to the rest of its body?\"\n[output]=\"Is the body of this dog breed large compared to its legs?\"\n\n[input]=\"Does this dog breed have short legs compared to the rest of its body?\"\n[output]=\"Are the legs of this dog breed proportionally shorter than its body?\"\n\n[input]=\"Does this dog breed have an average life expectancy range that can be more than 12 years?\"\n[output]=\"Does this dog breed have a lifespan of more than 12 years?\"\n\n",
    "task1622" : "[input]=\"Why was uh where was the Rhine regulated with an upper canal?\"\n[output]=\"Where was the Rhine regulated with an upper canal?\"\n\n[input]=\"What kind of committee considered legislation on the development of the Scottish or no make that the Edinburgh Tram Network?\"\n[output]=\"What kind of committee considered legislation on the development of the Edinburgh Tram Network?\"\n\n[input]=\"When degradation no economic inequality is smaller, more waste and pollution is?\"\n[output]=\"When economic inequality is smaller, more waste and pollution is?\"\n\n"
}

mean = {}
std = {}
range = {}
for task, example in examples.items():
    matches = re.findall(
        r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
        example,
        re.DOTALL,
    )
    outputs = []
    for match in matches:
        outputs.append(match[1])
    lengths = [len(s) for s in outputs]
    mean_length = np.mean(lengths)
    std_dev = np.std(lengths)
    mean[task] = mean_length
    std[task] = std_dev
    range[task] = (mean_length-2*std_dev, mean_length+2*std_dev)
print("mean")
print(mean)
print("std")
print(std)
print("range")
print(range)