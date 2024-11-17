## Location of the data:

Mock data: https://ldas-jobs.ligo.caltech.edu/~melissa.lopez/ML_course_mock/GlitchBank/mock_data/


### Data structure:

```
Numpy matrix: [samples, variables]
Samples: ['SNR', 'Chisq', 'Mass1', 'Mass2', 'Spin1z', 'Spin2z', 'Class']
```

Note that the data is boostrapped without replacement and the feature vector is averaged by signal-to-noise ration (SNR).
