# data=[[[308.19, 53.398, 0.84024], [217.7, 132.7, 0.76436], [249.73, 140.19, 0.71926], [219.57, 246.01, 0.79121], [0, 0, 0], [187.48, 128.9, 0.70718]]]
# x1 = 1
# y1 = 2
#
# data = [[[x[0]+x1,x[1]+y1,x[2]] if x[0]!=0 and x[1]!=0 else x for x in y] for y in data]
# #
# # for i in range(len(data)):
# #     for j in range(len(data[i])):
# #         if data[i][j][0] != 0 and data[i][j][1] != 0:
# #             data[i][j][0] += x1
# #             data[i][j][1] += y1
#
# print(data)

import os


def merge_txt_files(path, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                with open(os.path.join(path, filename)) as infile:
                    outfile.write(infile.read())


if __name__ == '__main__':
    input = "runs/detect/labels/"
    output = "runs/detect/1"
    merge_txt_files(input,output)