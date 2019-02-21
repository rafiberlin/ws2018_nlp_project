# Presentation

### Motivation
- Reinforce the strength of more discriminative categories
- Distinguishing between polysemous words

### Problem Statement
1. Which categories are strong indicators of sentiment? How significant are the differences?
2. What are the optimal weighting schemes? 
3. What linguistic phenomena is not captured by POS categories? (negation)


1. Strong indicators: 
    2. So far: Emoticons, R+A
2. Optimal weighting scheme:
    3. So far:
    ({'V': 2, 'N': 1, 'R+A': 3, 'E': 5, 'DEFAULT': 0}, (0.6422003360716952, 0.5864575675896431, 0.48346413300288466))
3. Need to output results of the best performing model: to manually look through it and see how negation works
4. Look for papers that analyse role of emoticons in semantics and in syntax
5. Look for papers that compare adjectives and adverbs and their role in sentences