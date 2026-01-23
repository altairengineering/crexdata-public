# import sys
# sys.path.append("../exp_original/pyMCDS.py")
import sys
from pyMCDS import pyMCDS,Settings
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib.animation as manimation
import matplotlib
cm_dic = {6:"alive",100:"apoptotic",101:"necrotic"}
def main():
# filename is xml filename without the path of the output
    instance_folder = ''
    print(instance_folder)
    output_data = instance_folder + './output/'
    settings = Settings(xml_file='config/set_deb.xml',output_path=instance_folder)
    print( settings.data['variables'])
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title="gamiee", artist='Matplotlib',
                    comment='Oxygen concentration')
    writer = FFMpegWriter(fps=1, metadata=metadata)

    matplotlib.rc('xtick', labelsize=10) 
    matplotlib.rc('ytick', labelsize=10) 
    matplotlib.rc('figure' , figsize=[8,8])    
    fig = plt.figure()
    cbar = None
    mov = settings.data['user_parameters']['multiplicity_of_infection']
    init_conc = settings.data['variables']['oxygen']

    # cbar.set_label('Concentration')
    with writer.saving(fig, instance_folder+"virion_concentration.mp4", 1):
        for index in range(481):
            # 2401
            filename ='output%08u.xml' % index
            print(filename)
            mcds = pyMCDS(filename,output_data)
            timepoint = index*30
            # plotigncellscode



            # concentration n data
            oxygen = mcds.get_concentrations('virion')

            X,Y = mcds.get_2D_mesh();
            # 
            plt.clf()
            
            im = plt.contourf(X,Y,oxygen[:,:,0]);
            # plt.clim(-1,1)
            plt.axis('image')
            plt.ticklabel_format(style='plain', axis='both')
            
            fmt = '%1.2f'
            cbar = plt.colorbar(im, format = fmt)
            
            plt.title(instance_folder+' o2 initial condition: '+str(init_conc)+' MOV: '+str(mov)+" timepoint:"+str(timepoint))
            plt.savefig("output/concentrations/"+str(timepoint)+".png")
            plt.show()
            if index!=0:
                cbar.update_normal(plt.contourf(X, Y, oxygen[:, :, 0]))
            # cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
            # writer.grab_frame()
        # plotting cells vs time
    # df = pd.DataFrame()

    # for data in data_list:
    #     new_df = pd.DataFrame.from_dict(data, orient='index')
    #     df = pd.concat([df, new_df], axis=0)

    # df.rename(columns={'index': 'Time'}, inplace=True)
    # df = df.ffill()
    # value_counts = {}

    # for index, row in df.iterrows():
    #     value_counts[index] = {}
    #     for col in df.columns[1:]:
    #         cell_value = row[col]            
    #         if pd.isnull(cell_value):
    #             if 'null' not in value_counts[index]:
    #                 value_counts[index]['null'] = 0
    #             value_counts[index]['null'] += 1
    #         else:
    #             if cell_value not in value_counts[index]:
    #                 value_counts[index][cell_value] = 0
    #             value_counts[index][cell_value] += 1

    # # Convert the dictionary to a DataFrame for better visualization
    # value_counts_df = pd.DataFrame.from_dict(value_counts, orient='index').fillna(0).astype(int)

    # plt.figure(figsize=(10, 6))
    # # Plot each column as a line plot
    # for col in value_counts_df.columns:
    #     plt.plot(value_counts_df.index, value_counts_df[col], label=f'{cm_dic.get(col, col)}')
    # plt.xlabel('Time')
    # plt.ylabel('Count')
    # plt.title('Count of Cell Phases over Time O2 initial condition:%s MOI: %s' % (init_conc,mov))
    # plt.legend()
    # plt.savefig("pngs/"+instance_folder+"_cells.png")









# def main():
#     cm_dic = {6:"alive",100:"apoptotic",101:"necrotic"}
# # filename is xml filename without the path of the output
#     instance_folder = sys.argv[1]
#     print(instance_folder)
#     output_data = instance_folder + '/output/'
#     data_list = []
#     settings = Settings(output_path=instance_folder)

#     # define concerntration movie initialization
#     FFMpegWriter = manimation.writers['ffmpeg']
#     metadata = dict(title=instance_folder, artist='Matplotlib',
#                     comment='Oxygen concentration')
#     writer = FFMpegWriter(fps=1, metadata=metadata)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     plt.xlabel('x')
#     plt.ylabel('y')
#     scatter = ax.scatter([],[], [], [], cmap='viridis')
    
#     # cbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
#     # cbar.set_label('Concentration')
#     with writer.saving(fig, "writer_test.mp4", 100):
#         for index in range(0,10):
#             # 
#             filename ='output%08u.xml' % index
#             print(filename)
#             mcds = pyMCDS(filename,output_data)
#             data = {}
#             timepoint = index*6
#             ids = mcds.data['discrete_cells']['ID']
#             ct = mcds.data['discrete_cells']['cell_type']
#             cp = mcds.data['discrete_cells']['cycle_model']
#             target_indices = [i for i, cell_type in enumerate(ct) if cell_type == 1]
#             target_cell_phases = [cp[i] for i in target_indices]
#             epithelial_ids = [ids[i] for i in target_indices]
#             k = dict(zip(epithelial_ids,target_cell_phases))
#             data[timepoint] = k
#             data_list.append(data)


    


#     # plotting cells vs time
#     df = pd.DataFrame()

#     for data in data_list:
#         new_df = pd.DataFrame.from_dict(data, orient='index')
#         df = pd.concat([df, new_df], axis=0)

#     df.rename(columns={'index': 'Time'}, inplace=True)
#     df = df.ffill()
#     value_counts = {}

#     # for index, row in df.iterrows():
#     #     value_counts[index] = {}
#     #     for col in df.columns[1:]:
#     #         cell_value = row[col]            
#     #         if pd.isnull(cell_value):
#     #             if 'null' not in value_counts[index]:
#     #                 value_counts[index]['null'] = 0
#     #             value_counts[index]['null'] += 1
#     #         else:
#     #             if cell_value not in value_counts[index]:
#     #                 value_counts[index][cell_value] = 0
#     #             value_counts[index][cell_value] += 1

#     # Convert the dictionary to a DataFrame for better visualization
#     value_counts_df = pd.DataFrame.from_dict(value_counts, orient='index').fillna(0).astype(int)
#     plt.figure(figsize=(10, 6))
#     # Plot each column as a line plot
#     for col in value_counts_df.columns:
#         plt.plot(value_counts_df.index, value_counts_df[col], label=f'{cm_dic.get(col, col)}')
#     plt.xlabel('Time')
#     plt.ylabel('Count')
#     plt.title('Count of Cell Phases over Time O2 initial condition:%s MOI: %s' % (settings.data["variables"]["oxygen"],settings.data['user_parameters']['multiplicity_of_infection']))
#     plt.legend()
#     plt.savefig("pngs/"+instance_folder+"_cells.png")


main()