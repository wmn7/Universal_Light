'''
@Author: WANG Maonan
@Date: 2022-06-21 22:33:40
@Description: 分析 sumo output 中的结果
@LastEditTime: 2022-07-08 20:03:25
'''
from xml.etree import ElementTree as ET
from aiolos.utils.get_abs_path import getAbsPath
pathConvert = getAbsPath(__file__)
import sys
sys.path.append(pathConvert('../'))

from SumoNets.NET_CONFIG import SUMO_NET_CONFIG as SUMO_CONFIG

def get_statistic_result(file_path):
    tree = ET.parse(file_path)

    for children in tree.iter():
        if children.tag=="vehicleTripStatistics":
            waiting_time = children.attrib["waitingTime"]
            return float(waiting_time)


if __name__ == '__main__':
    pathConvert = getAbsPath(__file__)

    net_folders = [
        'train_four_3', 'train_four_345', 'train_three_3', 
        'test_four_34', 'test_three_34', 'cologne1', 'ingolstadt1'
    ] # 所有测试的路网

    for _net_folder in net_folders:
        nets_name = SUMO_CONFIG[_net_folder]['nets'] # network file (包含多个路网文件)
        routes_name = SUMO_CONFIG[_net_folder]['routes'] # route file (包含多个车辆文件)

        for _net_name in nets_name:
            _net_name = _net_name.split('.')[0] # 获得路网名称
            # scnn 类模型
            scnn_results = list()
            scnn_all_results = list()
            # ernn 类模型
            ernn_results = list()
            ernn_all_results = list()
            # eattention 类模型
            eattention_results = list()
            eattention_all_results = list()

            for _route_name in routes_name:
                _route_name = _route_name.split('.')[0]
                # 文件路径
                scnn_path = pathConvert(f'./results/output/scnn/6_0_False_False_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                scnn_all_path = pathConvert(f'./results/output/scnn/6_0_True_True_True_True/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                ernn_path = pathConvert(f'./results/output/ernn/6_0_False_False_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                ernn_all_path = pathConvert(f'./results/output/ernn/6_0_True_True_True_True/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                eattention_path = pathConvert(f'./results/output/eattention/6_0_False_False_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                eattention_all_path = pathConvert(f'./results/output/eattention/6_0_True_True_True_True/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')

                # 分析文件结果
                scnn_wt = get_statistic_result(scnn_path)
                scnn_all_wt = get_statistic_result(scnn_all_path)
                ernn_wt = get_statistic_result(ernn_path)
                ernn_all_wt = get_statistic_result(ernn_all_path)
                eattention_wt = get_statistic_result(eattention_path)
                eattention_all_wt = get_statistic_result(eattention_all_path)

                # 添加文件结果
                scnn_results.append(scnn_wt)
                scnn_all_results.append(scnn_all_wt)
                ernn_results.append(ernn_wt)
                ernn_all_results.append(ernn_all_wt)
                eattention_results.append(eattention_wt)
                eattention_all_results.append(eattention_all_wt)

            print(
                f'{_net_folder}-{_net_name}:\n'
                f'SCNN: {scnn_results};\n'
                f'SCNN+ALL: {scnn_all_results};\n'
                f'ERNN: {ernn_results};\n'
                f'ERNN+ALL: {ernn_all_results};\n'
                f'EAttention: {eattention_results};\n'
                f'EAttention+ALL: {eattention_all_results};\n'
                f'---\n'
            )