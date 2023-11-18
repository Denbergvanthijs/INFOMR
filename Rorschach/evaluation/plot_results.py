import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from slugify import slugify

UU_YELLOW = "#FFCD00"
UU_RED = "#C00A35"
UU_CREME = "#FFE6AB"
UU_ORANGE = "#F3965E"
UU_BURGUNDY = "#AA1555"
UU_BROWN = "#6E3B23"
UU_BLUE = "#5287C6"
UU_PAL = sns.color_palette([UU_YELLOW, UU_RED, UU_CREME, UU_ORANGE, UU_BURGUNDY, UU_BROWN, UU_BLUE])


def plot_perclass_metrics(data_dict, metric, distance_function, k=None):
    labels, values = zip(*data_dict.items())
    # Sort alphabetically to make deterministic
    labels, values = zip(*sorted(zip(labels, values), key=lambda x: x[0]))

    k_str = k if k is not None else "|c|"
    metric_str = slugify(metric)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(labels, values, color=UU_YELLOW, width=0.7, edgecolor="black")
    plt.title(f"{metric.title()} per category for the {distance_function} distance function (k={k_str})")

    plt.xticks(rotation=90)
    plt.ylabel(metric.title())

    plt.xlim(-0.5, len(labels) - 0.5)      # Set x-limits to remove whitespace
    plt.ylim(0, 1)

    # Horizontal grid lines
    ax.set_axisbelow(True)  # Put grid behind bars
    ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig(f"./figures/step6/perclass/perclass_{metric_str}_k{k}_{distance_function}.png", dpi=300)
    # plt.show()


def plot_overall_metric(df, metric):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax = sns.barplot(x="Distance function", y=metric, data=df, ax=ax, palette=UU_PAL, edgecolor="black", hue="k")
    # plt.title(f"{metric.title()} per distance function")
    plt.ylabel(metric.title())

    # Change legend
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["k=1", "k=3", "k=|c|"], title="Top k meshes to consider", bbox_to_anchor=(1.05, 1))

    ax.set_axisbelow(True)  # Put grid behind bars
    ax.grid(axis="y")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"./figures/step6/overall_{metric.lower().replace(' ', '_')}.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    fp_results = "./Rorschach/evaluation/data/overall_results.csv"
    df = pd.read_csv(fp_results)
    print(df.head())

    plot_overall_metric(df, "Precision")
    plot_overall_metric(df, "Recall")
    plot_overall_metric(df, "Accuracy")
    plot_overall_metric(df, "Sensitivity")
    plot_overall_metric(df, "Specificity")
    plot_overall_metric(df, "F1 score")
    plot_overall_metric(df, "F2 score")

    # For debugging
    # results = {'Knife': 0.5308641975308642, 'Humanoid': 0.6549295774647887, 'Bicycle': 0.7564102564102563, 'Glasses': 0.7166666666666667, 'Door': 0.5606060606060606, 'AircraftBuoyant': 0.5416666666666666, 'Violin': 0.45614035087719296, 'Jet': 0.6969696969696969, 'ClassicPiano': 0.3684210526315789, 'Helicopter': 0.542483660130719, 'Gun': 0.5, 'Musical_Instrument': 0.42857142857142866, 'Truck': 0.375, 'Motorcycle': 0.46153846153846156, 'Shelf': 0.4806201550387597, 'Bus': 0.4444444444444444, 'Cup': 0.5888888888888888, 'Fish': 0.5333333333333333, 'HumanHead': 0.7466666666666666, 'Hand': 0.6756756756756758, 'House': 0.40579710144927533, 'Spoon': 0.4833333333333334, 'Biplane': 0.5555555555555555, 'PlantWildNonTree': 0.37037037037037046, 'Bookset': 0.4833333333333334, 'Train': 0.3452380952380953, 'Bottle': 0.4193548387096775, 'Bird': 0.45454545454545453, 'FloorLamp': 0.45238095238095244, 'Car': 0.5655430711610487, 'DeskLamp': 0.40860215053763443, 'Starship': 0.3978494623655915, 'DeskPhone': 0.38333333333333325, 'MultiSeat': 0.5185185185185185, 'Tree': 0.4113475177304963,
    #            'Cellphone': 0.4333333333333333, 'Ship': 0.5132275132275131, 'Tool': 0.5757575757575757, 'Computer': 0.5104166666666666, 'RectangleTable': 0.4791666666666667, 'Quadruped': 0.4912280701754385, 'Bed': 0.5308641975308642, 'PianoBoard': 0.4385964912280702, 'Insect': 0.4595959595959595, 'ComputerKeyboard': 0.7368421052631579, 'Wheel': 0.41176470588235287, 'NonWheelChair': 0.48333333333333334, 'SubmachineGun': 0.4833333333333334, 'Chess': 0.7301587301587302, 'Monoplane': 0.3833333333333333, 'BuildingNonResidential': 0.36363636363636365, 'Sword': 0.6792452830188679, 'PlantIndoors': 0.4974358974358974, 'Rocket': 0.45, 'AquaticAnimal': 0.4482758620689655, 'City': 0.44444444444444453, 'Guitar': 0.4833333333333333, 'MilitaryVehicle': 0.45, 'Monitor': 0.3833333333333333, 'Sign': 0.4444444444444443, 'TruckNonContainer': 0.41176470588235287, 'Mug': 0.4487179487179487, 'Apartment': 0.39130434782608703, 'Hat': 0.39999999999999997, 'WheelChair': 0.3833333333333333, 'RoundTable': 0.4594594594594595, 'Drum': 0.3833333333333333, 'Skyscraper': 0.38333333333333325, 'Vase': 0.39999999999999997}
    # plot_perclass_metrics(results, "F2 score", "EMD only", 3)
