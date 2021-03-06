var_label_dict = {
    'Race' : {0 : 'White', 
              1 : 'Black', 
              2 : 'Hispanic', 
              3 : 'Asian', 
              4 : 'Other'},
    'Gender' : {0 : 'Male', 
                1 : 'Female'},
    # 'Education0' : {0 : 'Less than High School', 
    #                1 : 'High School', 
    #                2 : 'Some College', 
    #                3 : 'Bachelor\'s', 
    #                4 : 'Advanced Degree'},
    'LegalStatus' : {0 : 'Naturalized Citizen',
                     1 : 'Lawful Noncitizen',
                     2 : 'Unauthorized Immigrant',
                     3 : 'Native-Born Citizen'},
    # 'ClassOfWorker' : {0 : 'Non-Worker',
    #                    1 : 'Self-Employed, Not Incorporated',
    #                    2 : 'Self-Employed, Incorporated', 
    #                    3 : 'Employed, Private Sector',
    #                    4 : 'Employed, Federal Government',
    #                    5 : 'Employed, State or Local Government'}, 
    'ClassOfWorker' : {0 : 'Non-Worker',
                       1 : 'Self-Employed',
                       2 : 'Private Sector', 
                       3 : 'Public Sector'}, 
    'Married' : {0 : 'Not Married', 
                 1 : 'Married'}, 
    'Year' : {2000 : '2000', 
              2015 : '2015', 
              2030 : '2030',
              2045 : '2045'}, 
    'Education' : {0 : 'Less than HS', 
                    1 : 'HS or Some College', 
                    2 : 'BA or Advanced Degree'}
}


slice_dict = {'Simple Means' : {
        'NaturalizedThisYear' : {'groups' : ['Gender', 'Married', 'Year'],
                                'condition_var' : 'LegalStatus',
                                'condition_isin' : [0], 
                                'condition_greater' : 'None'},
        'OverstayedVisaThisYear' : {'groups' : ['Gender', 'Married', 'Year'],
                                'condition_var' : 'LegalStatus',
                                'condition_isin' : [2], 'condition_greater' : 'None'}, 
        'EmigratedThisYear' : {'groups' : ['Year', 'Gender', 'Married'], 
                               'condition_var' : 'LegalStatus', 
                               'condition_isin' : [1, 2], 
                               'condition_greater' : 'None'},
        'Married' : {'groups' : ['Year', 'Education', 'Age'], 
                    'condition_var' : 'Age', 
                    'condition_greater' : 14, 
                    'condition_isin' : 'None'}, 
        'EducationYears' : {'groups' : ['Year', 'Gender', 'Age'], 
                            'condition_greater' : 'None', 
                            'condition_isin' : 'None', 
                            'condition_var' : 'None'},
        'DiedThisYear' : {'groups' : ['Year', 'Age', 'Gender'], 
                          'condition_greater' : 'None', 
                          'condition_isin' : 'None', 
                          'condition_var' : 'None'},
        'GaveBirthThisYear' : {'groups' : ['Year', 'Married', 'Age'], 
                               'condition_greater' : 'None', 
                               'condition_isin' : 'None', 
                               'condition_var' : 'None'},
        'WorkDisability' : {'groups' : ['Year', 'Age', 'Married'], 
                            'condition_greater' : 'None', 
                            'condition_isin' : 'None', 
                            'condition_var' : 'None'}, 
        'Worker' : {'groups' : ['Year', 'Education', 'Gender'], 
                    'condition_greater' : 'None', 
                    'condition_isin' : range(21, 55), 
                    'condition_var' : 'Age'}, 
        'ChildrenUnder18' : {'groups' : ['Age', 'Married', 'Year'], 
                             'condition_greater' : 'None', 
                             'condition_isin' : 'None', 
                             'condition_var' : 'None'}}, 
    'Categorical' : {'Race' : {'groups' : ['Gender', 'Year']}, 
                     'LegalStatus' : {'groups' : ['Year', 'Married']}, 
                     'ClassOfWorker' : {'groups' : ['Year', 'Gender']}}, 
    'Workers' : {'groups' : ['Year', 'Worker', 'ClassOfWorker', 'HoursWorked', 'PartTime', 'PartYear', 'logCoreWage', 'logCoreWageBase', 'logCoreWageShock', 'HourlyWage', 'WageIncome', 'Age', 'Education', 'Gender', 'Race', 'Married'], 'condition_var' : 'Worker', 'condition_isin' : [1], 'condition_greater' : 'None'}
}


chart_dict = {'Categorical' : {
                'Race' : {'facet' : 'Gender', 'ymax' : 0.8}, 
                'LegalStatus' : {'facet' : 'Married', 'ymax' : 1}, 
                'ClassOfWorker' : {'facet' : 'Gender', 'ymax' : .7}
}, 
              'Simple Means' : {
                  'NaturalizedThisYear' : {'xvar' : 'Year', 
                   'cat' : 'Married', 
                   'facet' : 'Gender', 
                   'units' : 'Percent of Eligible Population', 
                   'ymax' : 0.09, 
                   'smoothing' : 7},
                  'OverstayedVisaThisYear' : {'xvar' : 'Year', 
                   'cat' : 'Married', 
                   'facet' : 'Gender', 
                   'units' : 'Percent of Eligible Population', 
                   'ymax' : 0.11, 
                   'smoothing' : 7},
                  'EmigratedThisYear' : {'xvar' : 'Year', 
                                         'cat' : 'Married', 
                                         'facet' : 'Gender', 
                                         'units' : 'Percent of Eligible Population', 
                                         'ymax' : 0.05, 
                                         'smoothing' : 9}, 
                  'Married' : {'xvar' : 'Age',       
                               'cat' : 'Education', 
                               'facet' : 'Year', 
                               'units' : 'Percent of Sub-Group', 
                               'ymax' : .9, 
                               'smoothing' : 5}, 
                  'EducationYears' : {'xvar' : 'Age', 
                                      'cat' : 'Year', 
                                      'facet' : 'Gender', 
                                      'units' : 'Years of Education', 
                                      'ymax' : 16, 
                                      'smoothing' : 7}, 
                  'DiedThisYear' : {'xvar' : 'Age', 
                                    'cat' : 'Gender', 
                                    'facet' : 'Year', 
                                    'units' : 'Percent of Sub-Group', 
                                    'ymax' : 0.03, 
                                    'smoothing' : 7}, 
                  'GaveBirthThisYear' : {'xvar' : 'Age',
                                         'cat' : 'Married', 
                                         'facet' : 'Year', 
                                         'units' : 'Percent of Eligible Sub-Group', 
                                         'ymax' : 0.3, 
                                         'smoothing' : 7},
                  'WorkDisability' : {'xvar' : 'Age', 
                                      'cat' : 'Married', 
                                      'facet' : 'Year', 
                                      'units' : 'Percent of Sub-Group', 
                                      'ymax' : 0.4, 
                                      'smoothing' : 4}, 
                  'Worker' : {'xvar' : 'Year', 
                              'cat' : 'Gender', 
                              'facet' : 'Education', 
                              'units' : 'Percent of Sub-Group', 
                              'ymax' : 1, 
                              'smoothing' : 5}, 
                  'ChildrenUnder18' : {'xvar' : 'Age', 
                                       'cat' : 'Year', 
                                       'facet' : 'Married', 
                                       'units' : 'Percent of Sub-Group', 
                                       'ymax' : 1.75, 
                                       'smoothing' : 6}
              },
              'Workers' : {}}