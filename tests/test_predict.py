import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
from classifier import predict
from predict import make_prediction
from classifier.processing.datamanagement import load_dataset
# from processing import datamanagement
# from datamanagement import load_dataset


# test_make_single_prediction for predicting the labels of the inputs. the inputs should be called one by one
def test_make_single_prediction():

    test_data = load_dataset('interaction_360_for_labeling_15Jan20_AK.csv')
    single_test_input = test_data[0:1]
    results = make_prediction(input_data=single_test_input)
    # print(subject.get("predictions"))
    print(results)
    f = open("results.txt", 'a+')
    f.write(results["version"])
    f.write("\n")
    f.write("0= No, 1=Yes")
    f.write("\n")
    re = str(results["predictions"]).replace("[", "")
    re = re.replace("]", " ")
    f.write(re)

test_make_single_prediction()


