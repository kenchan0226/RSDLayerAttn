import matplotlib.pyplot as plt
import pandas as pd

def plt_select(select_id):
    if select_id == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2)
        data_uniter = {
            'Regions with IoU>0': [0.01349388704314784, 0.019055630969829023, 0.02108130148207597, 0.02481552055803821,
                                   0.028846489914473267,
                                   0.030598690965412578, 0.04200255819712182, 0.046710792960271745, 0.04390186562334256,
                                   0.04584466049089177,
                                   0.06720014109869077, 0.19682183684523197, 0.41962662269220513],
            'Regions with IoU=0': [0.0020323850031361015, 0.0029907052717168332, 0.0032147971734636607,
                                   0.0039490832190097564,
                                   0.004341250948190851, 0.004095712475786843, 0.0045697299158727, 0.005607450902942243,
                                   0.007355483879997827,
                                   0.009464041978658267, 0.022473306471827426, 0.8391393288162868, 0.09076672404088645]
            }
        df_uniter = pd.DataFrame(data_uniter, columns=['Regions with IoU>0', 'Regions with IoU=0'],
                          index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        # y_ticks = np.arange(13)
        # color = ["red", "blue"]
        df_uniter.plot.barh(ax=ax1, width=0.7)
        ax1.set_title("RSD-UNITER", fontsize=14)
        ax1.set_ylabel('Layer', fontsize=13)
        ax1.set_xlabel('Avg. attention score', fontsize=13)
        ax1.legend(prop={'size': 12})

        data_lxmert = {
            'Regions with IoU>0': [0.09405987053061461, 0.03614886241478976, 0.027972468885789133, 0.028925483895508836,
                                   0.03317450669970486,
                                   0.045600550913684514, 0.0518512942904102, 0.056335654038482105, 0.10786844741314593,
                                   0.11313489966126516,
                                   0.4049279606053009],
            'Regions with IoU=0': [0.015522085008864457, 0.004891094292848461, 0.003985808415923086,
                                   0.0039893113587423186,
                                   0.004634671823003102,
                                   0.006613228338671762, 0.008315189757049183, 0.010939890454485584,
                                   0.021262778066156476,
                                   0.03909457131567206,
                                   0.8807513708685616]
            }
        df_lxmert = pd.DataFrame(data_lxmert, columns=['Regions with IoU>0', 'Regions with IoU=0'],
                          index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # y_ticks = np.arange(13)
        # color = ["red", "blue"]
        df_lxmert.plot.barh(ax=ax2, width=0.7)
        ax2.set_title("RSD-LXMERT", fontsize=14)
        ax2.set_ylabel('Layer', fontsize=13)
        ax2.set_xlabel('Avg. attention score', fontsize=13)
        ax2.legend(prop={'size': 12})

        plt.savefig("figs/uniter_lxmert_layer_att_score_avg.pdf")
        plt.show()

    elif select_id == 'UNITER':
        data = {'Regions with IoU>0': [0.01349388704314784, 0.019055630969829023, 0.02108130148207597, 0.02481552055803821,
                  0.028846489914473267,
                  0.030598690965412578, 0.04200255819712182, 0.046710792960271745, 0.04390186562334256,
                  0.04584466049089177,
                  0.06720014109869077, 0.19682183684523197, 0.41962662269220513],
                'Regions with IoU=0': [0.0020323850031361015, 0.0029907052717168332, 0.0032147971734636607, 0.0039490832190097564,
                  0.004341250948190851, 0.004095712475786843, 0.0045697299158727, 0.005607450902942243,
                  0.007355483879997827,
                  0.009464041978658267, 0.022473306471827426, 0.8391393288162868, 0.09076672404088645]
                }
        df = pd.DataFrame(data, columns=['Regions with IoU>0', 'Regions with IoU=0'],
                          index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        #y_ticks = np.arange(13)
        #color = ["red", "blue"]
        df.plot.barh(width=0.7)
        plt.title("RSD-UNITER", fontsize=14)
        plt.ylabel('Layer', fontsize=13)
        plt.xlabel('Avg. attention score', fontsize=13)
        plt.legend(prop={'size': 12})
        plt.savefig("figs/uniter_layer_att_score_avg.pdf")
        plt.show()
    elif select_id == 'LXMERT':
        data = {'Regions with IoU>0': [0.09405987053061461, 0.03614886241478976, 0.027972468885789133, 0.028925483895508836,
                  0.03317450669970486,
                  0.045600550913684514, 0.0518512942904102, 0.056335654038482105, 0.10786844741314593,
                  0.11313489966126516,
                  0.4049279606053009],
                'Regions with IoU=0': [0.015522085008864457, 0.004891094292848461, 0.003985808415923086, 0.0039893113587423186,
                  0.004634671823003102,
                  0.006613228338671762, 0.008315189757049183, 0.010939890454485584, 0.021262778066156476,
                  0.03909457131567206,
                  0.8807513708685616]
                }
        df = pd.DataFrame(data, columns=['Regions with IoU>0', 'Regions with IoU=0'],
                          index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        #y_ticks = np.arange(13)
        #color = ["red", "blue"]
        df.plot.barh(width=0.7)
        plt.title("RSD-LXMERT", fontsize=14)
        plt.ylabel('Layer', fontsize=13)
        plt.xlabel('Avg. attention score', fontsize=13)
        plt.legend(prop={'size': 12})
        plt.savefig("figs/lxmert_layer_att_score_avg.pdf")
        plt.show()
    elif select_id == 'UNITER_IOU>0':
        plt.barh([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 [0.01349388704314784, 0.019055630969829023, 0.02108130148207597, 0.02481552055803821,
                  0.028846489914473267,
                  0.030598690965412578, 0.04200255819712182, 0.046710792960271745, 0.04390186562334256,
                  0.04584466049089177,
                  0.06720014109869077, 0.19682183684523197, 0.41962662269220513], align='center')
        plt.ylabel('UNITER encoder layer')
        plt.xlabel('Averaged attention score')
        plt.savefig("uniter_layer_iou_larger0_att_score_avg.pdf")
        plt.show()
    elif select_id == 'UNITER_IOU=0':
        plt.barh([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 [0.0020323850031361015, 0.0029907052717168332, 0.0032147971734636607, 0.0039490832190097564,
                  0.004341250948190851, 0.004095712475786843, 0.0045697299158727, 0.005607450902942243,
                  0.007355483879997827,
                  0.009464041978658267, 0.022473306471827426, 0.8391393288162868, 0.09076672404088645], align='center',
                 label="avg_data")
        plt.ylabel('UNITER encoder layer')
        plt.xlabel('Averaged attention score')
        plt.savefig("uniter_layer_iou_less0_att_score_avg.pdf")
        plt.show()
    elif select_id == 'LXMERT_IOU>0':
        plt.barh([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [0.09405987053061461, 0.03614886241478976, 0.027972468885789133, 0.028925483895508836,
                  0.03317450669970486,
                  0.045600550913684514, 0.0518512942904102, 0.056335654038482105, 0.10786844741314593,
                  0.11313489966126516,
                  0.4049279606053009], align='center')
        plt.ylabel('LXMERT encoder layer')
        plt.xlabel('Averaged attention score')
        plt.savefig("lxmert_layer_iou_larger0_att_score_avg.pdf")
        plt.show()
    elif select_id == 'LXMERT_IOU=0':
        plt.barh([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [0.015522085008864457, 0.004891094292848461, 0.003985808415923086, 0.0039893113587423186,
                  0.004634671823003102,
                  0.006613228338671762, 0.008315189757049183, 0.010939890454485584, 0.021262778066156476,
                  0.03909457131567206,
                  0.8807513708685616], align='center')
        plt.ylabel('LXMERT encoder layer')
        plt.xlabel('Averaged attention score')
        plt.savefig("lxmert_layer_iou_less0_att_score_avg.pdf")
        plt.show()
    else:
        print("Input errors, please enter again")

if __name__ == "__main__":
    #select_id = "UNITER"
    select_id = "LXMERT"
    #select_id = "both"
    plt_select(select_id)
