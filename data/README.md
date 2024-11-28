## Location of the data:

Mock data: https://ldas-jobs.ligo.caltech.edu/~melissa.lopez/ML_course_mock/GlitchBank/mock_data/


### Data structure:

```
Numpy matrix: [samples, variables]
Samples: ['SNR', 'Chisq', 'Mass1', 'Mass2', 'Spin1z', 'Spin2z', 'Class']
```

Note that the data is boostrapped without replacement and the feature vector is averaged by signal-to-noise ration (SNR).

```
def classTag(file_name):
    if 'Injections' in file_name:
        t = 0
    if 'Blip' in file_name:
        t = 1
    if 'Koi_Fish' in file_name:
        t = 2
    if 'Low_Frequency_Burst' in file_name:
        t = 3
    if 'Tomte' in file_name:
        t = 4
    if 'Whistle' in file_name:
        t = 5
    if 'Fast_Scattering' in file_name:
        t = 6
    return t
```
WARNING: Note that Virgo does NOT contain 'Fast_Scattering' (class 6)
