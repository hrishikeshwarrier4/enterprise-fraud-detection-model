# Example transform - uncomment to use
# from transforms.api import Input, Output, transform
#
# @transform.using(
#     output_dataset=Output("/path/to/output"),
#     input_dataset=Input("/path/to/input"),
# )
# def compute(input_dataset, output_dataset):
#     output_dataset.write_table(input_dataset.polars(lazy=True))
