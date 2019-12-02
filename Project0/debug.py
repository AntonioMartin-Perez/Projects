def get_sum_metrics(predictions, metrics=()):
    for i in range(3):
        def f(x,i=i): return x+i
        metrics +=(f,)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)
    return sum_metrics


def main():
    print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9
    print(get_sum_metrics(3, [lambda x: x]))  # Should be (3) + (3 + 0) + (3 + 1) + (3 + 2) = 15
    print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9

if __name__ == "__main__":
    main()

"""To summarize: metrics is a list of functions that contains at 
least the functions  f0(x)=x,f1(x)=x+1,f2(x)=x+2 . It also can contain 
additional functions  f3(x),f4(x),...  that the user (optionally) gives 
in as argument to the function as a list  [f3,...,fn] . The function 
get_sum_metrics should then output  ∑ifi(prediction).

"""