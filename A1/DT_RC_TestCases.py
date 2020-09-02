from src.main import *
import pandas as pd


def test_case():
    # outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(',')
    # temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(',')
    # humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(',')
    # windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE'.split(',')
    # play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')
    fever = 'no,yes,yes,yes,yes,no,yes,yes,no,yes,no,no,no,yes'.split(',')
    cough = 'no,yes,yes,no,yes,yes,no,no,yes,yes,yes,yes,yes,yes'.split(',')
    breathing_issues = 'no,yes,no,yes,yes,no,yes,yes,yes,no,no,yes,yes,no'\
        .split(',')
    infected = 'no,yes,no,yes,yes,no,yes,yes,yes,yes,no,no,yes,no'.split(',')

    fever = 'yes,yes,yes,yes,yes'.split(',')
    cough = 'yes,no,yes,no,no'.split(',')
    breathing_issues = 'yes,yes,yes,yes,yes'.split(',')
    infected = 'yes,yes,yes,yes,yes'.split(',')


    fever = 'no,no,no,no,no,yes,no,yes,no'.split(',')
    cough = 'yes,yes,yes,yes,no,yes,yes,no,yes'.split(',')
    breathing_issues = 'yes,yes,yes,no,no,yes,yes,yes,no'.split(',')
    infected = 'yes,no,no,no,no,yes,no,no,yes'.split(',')

    fever = 'no,no,no,no,no,yes,no,yes,no'.split(',')
    cough = 'yes,yes,yes,yes,no,yes,yes,no,yes'.split(',')
    breathing_issues = 'yes,yes,yes,no,no,yes,yes,yes,no'.split(',')
    infected = 'yes,no,no,no,no,yes,no,no,no'.split(',')

    fever = 'no,no,no,no,no,yes,no,yes,no,no,yes,no'.split(',')                                                                                             
    cough = 'yes,yes,yes,yes,no,yes,yes,no,yes,yes,no,yes'.split(',')                                                                                       
    breathing_issues = 'yes,yes,yes,no,no,yes,yes,yes,no,yes,yes,no'.split(',')                                                                                        
    infected = 'yes,no,maybe,no,no,yes,maybe,no,maybe,maybe,no,maybe'.split(',')   

    # fever = 'no,no,no'.split(',')
    # cough = 'yes,yes,yes'.split(',')
    # breathing_issues = 'yes,yes,yes'.split(',')
    # infected = 'yes,no,no'.split(',')


    columns = [fever, cough, breathing_issues, infected]

    # sanity check
    if not (len(fever) == len(cough) and len(breathing_issues) == len(infected) and
       len(fever) == len(breathing_issues)):
        print("Columnn not of equal length!")
        for column in columns:
            print(len(column))
        return 1

    dataset = {'fever': fever, 'cough': cough,
               'breathing_issues': breathing_issues,
               'infected': infected}

    df = pd.DataFrame(dataset, columns=['fever', 'cough', 'breathing_issues',
                      'infected'])

    try:
        if get_entropy_of_dataset(df) >= 0.985 and \
           get_entropy_of_dataset(df) <= 0.986:
            print("Test Case 1 for the function get_entropy_of_dataset PASSED")
        else:
            print("Test Case 1 for the function get_entropy_of_dataset FAILED with",
                  get_entropy_of_dataset(df))
    except:
        print("Test Case 1 for the function get_entropy_of_dataset FAILED")

    try:
        if get_entropy_of_attribute(df, 'fever') >= 0.852 and \
           get_entropy_of_attribute(df, 'fever') <= 0.862 :
            print("Test Case 2 for the function get_entropy_of_attribute PASSED")
        else:
            print("Test Case 2 for the function get_entropy_of_attribute FAILED", get_entropy_of_attribute(df, 'fever'))

    except:
         print("Test Case 2 for the function get_entropy_of_attribute FAILED")

    try:
        if get_entropy_of_attribute(df, 'cough') >= 0.945 and \
           get_entropy_of_attribute(df, 'cough') <= 0.947:
            print("Test Case 3 for the function get_entropy_of_attribute PASSED")
        else:
            print("Test Case 3 for the function get_entropy_of_attribute FAILED", get_entropy_of_attribute(df, 'cough'))

    except:
        print("Test Case 3 for the function get_entropy_of_attribute FAILED")

    # try:
    #     # columns=['fever', 'cough', 'breathing_issues', 'infected']
    ans = get_selected_attribute(df)
    #     dictionary = ans[0]
    #     flag = (dictionary['fever'] >= 0.244 and dictionary['fever'] <= 0.248)\
    #         and\
    #            (dictionary['cough'] >= 0.0292 and dictionary['cough'] <= 0.0296)\
    #         and\
    #            (dictionary['breathing_issues'] >= 0.150 and dictionary['breathing_issues'] <= 0.154)\
    #         and\
    #            (ans[1] == 'fever')

    print(ans)

    #     if flag:
    #         print("Test Case 4 for the function get_selected_attribute PASSED")
    #     else:
    #         print("Test Case 4 for the function get_selected_attribute FAILED")
    # except:
    #     print("Test Case 4 for the function get_selected_attribute FAILED")


if __name__ == "__main__":
    test_case()
