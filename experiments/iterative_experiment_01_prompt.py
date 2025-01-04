
import random

# Let's have 2-shot prompting for this
prompt1 = [
    "My lawn measures 30 feet by 40 feet. How many square feet is it?",
    "30 feet * 40 feet = 1200 square feet. #### 1200",
]

prompt2 = [
    "SpaceX wishes to launch an average of 20 rockets per month over this next year. In the first 7 months of the year, they launched 30, 20, 40, 50, 10, 20, and 30 rockets. How many rockets-per-month do they need to launch in the last 5 months of the year to meet their goal?",
    "In the first 4 months, that's 30 + 20 + 40 + 50 = 140 rockets. In the next 3 months, they launched 10 + 20 + 30 = 60 rockets. So in the first 7 months, they launched 200 rockets. To launch an average of 20 rockets per month over the year, they need to launch 20 * 12 = 240 rockets over the year. So they need to launch 240 - 200 = 40 rockets in the last 5 months, or 40 / 5 = 8 rockets per month. #### 8",
]

prompt3 = [
    "The Greeks wish to send 1000 ships to Troy. If everyone apart from Menelaus, Odysseus, and Diomedes have already supplied 820 ships, how many do each of these three need to supply to meet the goal?",
    "The Greeks need 1000 ships. 820 ships have already been supplied. So they need 1000 - 820 = 180 ships. These three need to supply 180 / 3 = 60 ships each. #### 60",
]

prompt4 = [
    "Steve feels self-loathing on average 50% of the time after his breakup with his girlfriend. After the breakup, he spent 1 week eating ice cream in his basement, 2 weeks on a trip to Vietnam, and an addition 13 days working late nights to lose himself in work. How many days did he spend self-loathing?",
    "Steve spent 1 week in his basement, 2 weeks in Vietnam, and an addition 13 days working late nights to lose himself in work. That's 7 + 14 + 13 = 34 days. He feels self-loathing on average 50% of the time, so he spent 34 * 0.5 = 17 days self-loathing. #### 17",
]


def get_random_prompts(k = 2):
    sampled = random.sample([prompt1, prompt2, prompt3, prompt4], k)
    
    # coonvert to "role" and "content" format
    sampled = [
        [{"role": "user", "content": prompt[0]}, {"role": "assistant", "content": prompt[1]}]
        for prompt in sampled
    ]

    # flatten the list
    sampled = [item for sublist in sampled for item in sublist]

    return sampled



