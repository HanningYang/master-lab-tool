import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
import math
from scipy.stats import lognorm
from oauth2client.service_account import ServiceAccountCredentials

params1 = []
if 'parameters_saved' not in st.session_state:
    st.session_state['parameters_saved'] = False


def find_empty_row(sheet):
    """Find the first empty row in Column A."""
    col_A = sheet.col_values(1)  # Get all values in column A
    return len(col_A) + 1  # Next available row index

def save_parameters(name, comment, parameters, add_variability, variability_level, birth_means_var):
    try:
        # Use credentials from Streamlit secrets
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            st.secrets["google_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        client = gspread.authorize(credentials)

        # Access the Google Sheet
        sheet = client.open("ODE-parameters").sheet1
        
        # Find the first empty row in Column A
        next_row = find_empty_row(sheet)

        # Prepare the data to append
        data = [name, comment] + parameters + [add_variability, variability_level] + birth_means_var
        
        # Insert data into the first empty row (starting from column A)
        sheet.insert_row(data, next_row)

        st.success(f"Parameters successfully saved in row {next_row}.")

    except gspread.exceptions.APIError as e:
        st.error(f"API Error: {e}")




### Function Definitions

def ode_system(y, t, K_C, K_H, K_W, K_A, K_I, r_C, r_H, r_W, r_A, r_I,
                alpha_HI, alpha_AW):
    C, H, W, A, I = y
    dCdt = r_C * C * (1 - C / K_C)
    dHdt = r_H * H * (1 - H / K_H) + alpha_HI * (I / K_I)
    dWdt = r_W * W * (1 - W / K_W)
    dAdt = r_A * A * (1 - A / K_A) + alpha_AW * (W / K_W)
    dIdt = r_I * I * (1 - I / K_I) 
    return np.array([dCdt, dHdt, dWdt, dAdt, dIdt])

def euler_method(func, y0, t, *args):
    ys = [y0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        y_prev = ys[-1]
        dydt = func(y_prev, t[i - 1], *args)
        y_new = y_prev + dydt * dt
        ys.append(y_new)
    solu = np.array(ys)
    return solu

def apply_noise(data, size, noise_std):
    noise_C = np.random.normal(0, noise_std[0], size)
    noise_H = np.random.normal(0, noise_std[1], size)
    noise_W = np.random.normal(0, noise_std[2], size)
    noise_A = np.random.normal(0, noise_std[3], size)
    noise_I = np.random.normal(0, noise_std[4], size)
    noise = np.stack([noise_C, noise_H, noise_W, noise_A, noise_I], axis=1)
    data_with_noise = data + noise
    # Adjust noise to ensure no negative values in data_with_noise
    negative_indices = data_with_noise < 0
    abs_noise = np.abs(noise)
    data_with_noise[negative_indices] = data_with_noise[negative_indices] - noise[negative_indices] + abs_noise[negative_indices]

    return data_with_noise

def process_data(data, size, noise, noise_std):
    if noise:
        data = apply_noise(data.copy(), size, noise_std)
    return data


def concatenate_data_diff_noise(init, time_points, maxes, cali_params, noise_std, noise, type='hypo', inter_num=None):
    if type == 'real':
        if inter_num is None:
            raise ValueError("inter_num must be provided when type is 'real'")
        inter_time_points = time_points[:inter_num]
        se_time_points = time_points[inter_num:]        
    else:
        mid_index = len(time_points) // 2
        inter_time_points = time_points[:mid_index]
        se_time_points = time_points[mid_index:]

    final_data = []
    K_C, K_H, K_W, K_A, K_I = maxes
    r_C, r_H, r_W, r_A, r_I, alpha_HI, alpha_AW = cali_params

    for idx, t in enumerate(time_points):
        t = np.array(t)
        y0 = init[:, idx]
        inter = euler_method(ode_system, y0, t, K_C, K_H, K_W, K_A, K_I,
                             r_C, r_H, r_W, r_A, r_I,
                             alpha_HI, alpha_AW)
        size = inter.shape[0]
        inter = process_data(inter, size, noise, noise_std)
        t = t.reshape(-1, 1)
        inter = np.hstack([inter, t])
        final_data.append(inter)

    return final_data

def resample_positive(mean, std, size):
    values = np.random.normal(loc=mean, scale=std, size=size)
    while (values < 0).any():
        negative_indices = values < 0
        values[negative_indices] = np.random.normal(loc=mean, scale=std, size=negative_indices.sum())
    return values

def generate_initial_conditions(num_patient, mean_C, std_C, mean_H, mean_W, mean_A, mean_I):

    sigma_sq = math.log(1 + (std_C / mean_C)**2)
    mu = math.log(mean_C) - (sigma_sq / 2)
    sigma = math.sqrt(sigma_sq)
    C_simu = lognorm.rvs(s=sigma, scale=np.exp(mu), size=num_patient)
    
    H_simu = resample_positive(mean=mean_H, std=2.5, size=(num_patient,))
    W_simu = resample_positive(mean=mean_W, std=2.3, size=(num_patient,))
    A_simu = resample_positive(mean=mean_A, std=0.9, size=(num_patient,))
    I_simu = resample_positive(mean=mean_I, std=21, size=(num_patient,))
    
    simu_init = np.stack([C_simu, H_simu, W_simu, A_simu, I_simu])
    return simu_init


def plot_simulation(final_data):
    # Define colors for each biomarker
    colors = ['royalblue', 'mediumseagreen', 'salmon', 'gold', 'plum']
    # y_limits = [(0, 200), (0, 20), (5, 30), (0.0, 8.0), (0, 150)]
    biomarkers = ['CRP', 'Haemoglobin', 'BMI', 'Albumin', 'Iron']

    # Plot each patient’s data in separate figures
    num_patients = len(final_data)
    for idx, patient_data in enumerate(final_data):
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 30))
        time = patient_data[:, -1]
        
        # Plot each biomarker separately
        for i, ax in enumerate(axes[:5]):
            ax.plot(time, patient_data[:, i], label=biomarkers[i], color=colors[i])
            # ax.set_ylim(y_limits[i])
            ax.set_ylabel(biomarkers[i])
            ax.set_xlabel('Age')
            ax.legend()
        
        # Combined plot for all biomarkers in the last subplot
        for i, biomarker in enumerate(biomarkers):
            axes[5].plot(time, patient_data[:, i], label=biomarker, color=colors[i])
        axes[5].set_ylabel('Values')
        axes[5].set_xlabel('Age')
        axes[5].set_title('Combined Biomarkers')
        axes[5].legend()
        
        # Set labels and titles
        axes[-1].set_xlabel('Time')
        fig.suptitle(f'Patient {idx + 1} Progression')
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        st.pyplot(fig)

def generate_log_normal_samples(mean_C, std_C, num_samples=1000):
    sigma_sq = np.log(1 + (std_C / mean_C) ** 2)  # Log variance
    mu = np.log(mean_C) - (sigma_sq / 2)  # Log mean
    sigma = np.sqrt(sigma_sq)  # Log std dev
    return lognorm.rvs(s=sigma, scale=np.exp(mu), size=num_samples)


def plot_initial_conditions_distributions(mean_C, std_C, mean_H, mean_W, mean_A, mean_I):
    num_samples = 1000  # Number of samples to plot the distributions

    # Generate samples for each biomarker
    C_samples = generate_log_normal_samples(mean_C, std_C, num_samples)
    H_samples = np.random.normal(mean_H, 2.5, num_samples)
    W_samples = np.random.normal(mean_W, 2.3, num_samples)
    A_samples = np.random.normal(mean_A, 0.9, num_samples)
    I_samples = np.random.normal(mean_I, 21, num_samples)

    # Create subplots with two rows: first row for CRP and Hemoglobin, second row for BMI, Albumin, and Iron
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Initial Conditions Distributions", fontsize=16, weight='bold')

    # Plot each biomarker with appropriate spacing and axis limits
    sns.histplot(C_samples, kde=True, ax=axes[0, 0], color="royalblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("CRP Distribution")
    axes[0, 0].set_xlabel("CRP")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_xlim(0, 200)  # Adjusted x-axis limits for clarity

    sns.histplot(H_samples, kde=True, ax=axes[0, 1], color="mediumseagreen", edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Haemoglobin Distribution")
    axes[0, 1].set_xlabel("Haemoglobin")
    axes[0, 1].set_ylabel("Frequency")

    sns.histplot(W_samples, kde=True, ax=axes[1, 0], color="salmon", edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("BMI Distribution")
    axes[1, 0].set_xlabel("BMI")
    axes[1, 0].set_ylabel("Frequency")

    sns.histplot(A_samples, kde=True, ax=axes[1, 1], color="gold", edgecolor="black", alpha=0.7)
    axes[1, 1].set_title("Albumin Distribution")
    axes[1, 1].set_xlabel("Albumin")
    axes[1, 1].set_ylabel("Frequency")

    sns.histplot(I_samples, kde=True, ax=axes[1, 2], color="plum", edgecolor="black", alpha=0.7)
    axes[1, 2].set_title("Iron Distribution")
    axes[1, 2].set_xlabel("Iron")
    axes[1, 2].set_ylabel("Frequency")

    # Hide any unused subplot axes for a cleaner look
    axes[0, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit titles
    st.pyplot(fig)



# Set plot style
# plt.style.use('seaborn-darkgrid')
plt.style.use('ggplot')


st.title("RDEB Patient Progression Simulation")

st.markdown("""
Welcome to **RDEB Patient Progression Simulation App**. This app allows you to adjust simulation parameters and visualize the impact on RDEB patient outcomes.
""")


st.write("""
We simulate **5 biomarkers** in EB patients:
""")

st.markdown("""
- **C (CRP)** - Inflammatory marker  
- **H (Hemoglobin)** - Oxygen-carrying protein  
- **B (BMI)** - Body Mass Index  
- **A (Albumin)** - Protein linked to nutrition & liver function  
- **I (Iron)** - Essential mineral for blood function  
""")

st.header("ODE System for EB Biomarkers")

# Display the ODE System
st.latex(r"""
\begin{aligned}
\frac{dC}{dt} &= r_C C \left(1 - \frac{C}{K_C} \right) \\
\frac{dH}{dt} &= r_H H \left(1 - \frac{H}{K_H} \right) + \alpha_{HI} \left(\frac{I}{K_I} \right) \\
\frac{dW}{dt} &= r_W W \left(1 - \frac{W}{K_W} \right) \\
\frac{dA}{dt} &= r_A A \left(1 - \frac{A}{K_A} \right) + \alpha_{AW} \left(\frac{W}{K_W} \right) \\
\frac{dI}{dt} &= r_I I \left(1 - \frac{I}{K_I} \right)
\end{aligned}
""")

# Explanation with full LaTeX
with st.expander("📊 **Mathematical Explanation**"):
    st.markdown(r"""
    **Growth Terms:**
    
    $$ \frac{dX}{dt} = r_X X \left(1 - \frac{X}{K_X} \right) $$
    
    where \( X \in \{C, H, W, A, I\} \).
    
    - Each biomarker \( X \) follows a **logistic growth model**.  
    - The term \( r_X X \) represents **natural growth**.  
    - The factor \( \left(1 - \frac{X}{K_X} \right) \) ensures that growth slows  
      as \( X \) approaches its **carrying capacity** \( K_X \).  
    """)

    st.markdown(r"""
    **Interaction Terms:**
    
    $$ \alpha_{HI} \frac{I}{K_I}, \quad \alpha_{AW} \frac{W}{K_W} $$
    
    - Some biomarkers depend on others:  
      - \( H \) (Hemoglobin) receives an **additional boost** from \( I \) (Iron),  
        scaled by \( \alpha_{HI} \).  
      - \( A \) (Albumin) increases with \( W \) (Weight), controlled by \( \alpha_{AW} \).  
    """)


with st.expander("🧑‍🤝‍🧑 **Simple Explanation Using Math**"):
    st.markdown(r"""
    $$ \frac{dX}{dt} = \text{Growth} + \text{Influence from Others} $$

    - The equation $ \frac{dX}{dt} $ describes **how a biomarker changes over time**.  
    - Growth follows the rule:  
      $$ \frac{dX}{dt} = r_X X \left(1 - \frac{X}{K_X} \right) $$  
    - Growth is limited by the **carrying capacity** $ K_X $, ensuring that $ X $ does not grow indefinitely.

    - Some biomarkers are **linked**:  
      $$ \frac{dH}{dt} = r_H H \left(1 - \frac{H}{K_H} \right) + \alpha_{HI} \frac{I}{K_I} $$
      - Hemoglobin $ H $ grows when iron $ I $ is high.  
      $$ \frac{dA}{dt} = r_A A \left(1 - \frac{A}{K_A} \right) + \alpha_{AW} \frac{W}{K_W} $$
      - Albumin $ A $ grows when weight $ W $ is high.  
    """)




st.markdown("""
**Instructions:**
- Adjust the parameters in the sidebar, where each parameter includes a brief description. The default values serve as initial estimates.
- Click **"Run Simulation"** to update the results.
- Click **"Save Parameters"** to store the parameter settings you find reasonable.
""")


# --------------------------------------------------------------------------------------------------------------------------------------------------------

# Sidebar parameters
st.sidebar.title("Adjust Parameters")

# st.sidebar.markdown("### Simulation Parameters")


# Explanation for parameters
# st.sidebar.info("""
# # **Parameters:**

# - **Growth Rates (r_C, r_H, r_W, r_A, r_I)**: Control the growth rate of each cell type.
# - **Interaction Parameters (alpha, beta, theta)**: Define how different biomarkers influence each other.
# - **Noise Level**: Adjusts the variability in the simulation data.
# - **Initial Conditions**: Mean values for the initial quantities of each cell type.
# """)

st.sidebar.markdown("""
### Enter Initial Biomarker Values

Enter the **mean values** for hemoglobin, BMI, albumin, and iron,  
as well as **mean** and **standard deviation** for CRP.  

**Note:** Standard deviations for hemoglobin, BMI, albumin, and iron are **predefined**.
""")

with st.sidebar.expander("🔍 What Distributions Are Used?"):
    st.markdown("""
    - The initial conditions for hemoglobin, BMI, albumin, and iron follow **normal distributions**.  
    - CRP follows a **log-normal distribution**, meaning the **mean** and **standard deviation** you enter  
      are in the **original (linear) scale** and will be transformed into log-space parameters  
      before generating CRP samples.
    """)

# Collapsible explanation section
with st.sidebar.expander("🔍 What Does This Mean?"):
    st.markdown("""
    - The distributions specified here are used **to generate the initial conditions** of the biomarkers.  
    - The **mean_C** and **std_C** you enter for CRP are in the **original (linear) scale**.  
    - They will be **transformed into log-space parameters** before generating CRP samples.
    """)



# Initial conditions
# Intermediate Group inputs
st.sidebar.markdown("#### Mean of Biomarker")
mean_H_simu_inter = st.sidebar.number_input('Haemoglobin', value=10.6)
mean_W_simu_inter = st.sidebar.number_input('BMI', value=14.0)
mean_A_simu_inter = st.sidebar.number_input('Albumin', value=3.8)
mean_I_simu_inter = st.sidebar.number_input('Iron', value=32.5)
mean_C_simu_inter = st.sidebar.number_input('CRP', value=19.0)

st.sidebar.markdown("#### Standard Deviation of CRP")
std_C_inter = st.sidebar.number_input('CRP', value=29.0)


# Sidebar option to visualize distributions
if st.sidebar.checkbox("Visualize Initial Conditions Distributions"):
    st.markdown("## Initial Conditions Distributions")
    st.markdown("This section allows you to view the distribution of initial conditions based on the specified means and predefined standard deviations.")

    # Visualize intermediate group distribution
    plot_initial_conditions_distributions(
        mean_C=mean_C_simu_inter,
        std_C=std_C_inter,
        mean_H=mean_H_simu_inter,
        mean_W=mean_W_simu_inter,
        mean_A=mean_A_simu_inter,
        mean_I=mean_I_simu_inter
    )
    









# Parameters to adjust
st.sidebar.markdown("## Growth/Deacay Rate of Different Biomarkers")
st.sidebar.markdown("A positive value for the growth/decay rate indicates that the biomarker tends to increase (growth) over time, while a negative value indicates a decrease (decay) over time.")
r_C1 = st.sidebar.slider('CRP', min_value=-1.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
r_H1 = st.sidebar.slider('Haemoglobin', min_value=-1.0, max_value=1.0, value=-0.1, step=0.01, format="%.2f")
r_W1 = st.sidebar.slider('BMI', min_value=-1.0, max_value=1.0, value=-0.05, step=0.01, format="%.2f")
r_A1 = st.sidebar.slider('Albumin', min_value=-1.0, max_value=1.0, value=-0.1, step=0.01, format="%.2f")
r_I1 = st.sidebar.slider('Iron', min_value=-1.0, max_value=1.0, value=-0.1, step=0.01, format="%.2f")

st.sidebar.markdown("## How the First Biomarker Influences the Second One")
alpha_HI1 = st.sidebar.slider('Iron - Hemoglobin', min_value=-1.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")
alpha_AW1 = st.sidebar.slider('BMI - Albumin', min_value=-1.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")


st.sidebar.markdown("## Variability")
st.sidebar.markdown("If you choose not to add variability, you can leave the variability level alone.")
add_noise = st.sidebar.checkbox('Add Variability', True)
noise_level = st.sidebar.number_input('Variability Level', min_value=0.0, max_value=2.0, value=0.1, step=0.1, format="%.1f")







# Function to generate different time points per patient
def generate_patient_time_points(num_patients, max_time=15):
    time_points = []
    for _ in range(num_patients):
        num_time_steps = np.random.randint(5, 20)  # Random number of time points per patient
        time_points.append(sorted(np.random.uniform(0, max_time, num_time_steps)))
    return time_points


# Compute percentiles across all patients at each time step
def compute_percentiles(all_patient_data, time_grid, num_biomarkers=5):
    percentiles = {p: np.zeros((len(time_grid), num_biomarkers)) for p in [5, 50, 95]}

    for i, t in enumerate(time_grid):
        biomarker_values = [[] for _ in range(num_biomarkers)]

        # Collect values for each biomarker at time t
        for patient_data in all_patient_data:
            closest_idx = np.argmin(np.abs(patient_data[:, -1] - t))  # Closest time point
            for j in range(num_biomarkers):
                biomarker_values[j].append(patient_data[closest_idx, j])

        # Compute percentiles for each biomarker
        for j in range(num_biomarkers):
            percentiles[5][i, j] = np.percentile(biomarker_values[j], 5)
            percentiles[50][i, j] = np.percentile(biomarker_values[j], 50)
            percentiles[95][i, j] = np.percentile(biomarker_values[j], 95)

    return percentiles

# Plot five separate figures for each biomarker
def plot_biomarker_trajectories(all_patient_data, percentiles, time_grid):
    biomarkers = ['CRP', 'Haemoglobin', 'BMI', 'Albumin', 'Iron']
    colors = ['royalblue', 'mediumseagreen', 'salmon', 'gold', 'plum']

    fig, axes = plt.subplots(5, 1, figsize=(8, 25), sharex=True)
    fig.suptitle("ODE Simulation of Biomarkers with Percentiles", fontsize=16, weight='bold')

    for i, ax in enumerate(axes):
        # Scatter plot of all patient data
        for patient_data in all_patient_data:
            ax.scatter(patient_data[:, -1], patient_data[:, i], color=colors[i], alpha=0.3, s=15)

        # Plot percentiles
        ax.plot(time_grid, percentiles[5][:, i], linestyle="--", color="black", label="5th percentile")
        ax.plot(time_grid, percentiles[50][:, i], linestyle="-", color="black", label="50th percentile (Median)")
        ax.plot(time_grid, percentiles[95][:, i], linestyle="--", color="black", label="95th percentile")

        ax.set_title(biomarkers[i])
        ax.set_ylabel("Value")
        ax.legend()

    axes[-1].set_xlabel("Age (years)")  # Set x-axis label only on the last plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust spacing
    st.pyplot(fig)

# Automatically generate and display the plots when the app loads
num_patients = 20
max_time = 15

# Generate different time points per patient
time_points_simu = generate_patient_time_points(num_patients, max_time)

# Generate initial conditions for 20 patients
simu_init = generate_initial_conditions(
    num_patients, mean_C_simu_inter, std_C_inter,
    mean_H_simu_inter, mean_W_simu_inter, mean_A_simu_inter, mean_I_simu_inter
)

# Define ODE parameters
maxes = (200.0, 15.0, 25.0, 5.0, 160.0)
params = [r_C1, r_H1, r_W1, r_A1, r_I1, alpha_HI1, alpha_AW1]
noise_std = [40 * noise_level, 2.5 * noise_level, 2.3 * noise_level, 0.9 * noise_level, 21 * noise_level]

# Simulate patient data
final_data = concatenate_data_diff_noise(
        simu_init, time_points_simu,
        maxes, params, noise_std, noise=add_noise, type='hypo'
    )
# simulate_patient_data(num_patients, time_points_simu, simu_init, maxes, params, noise_std, add_noise)

# Create a common time grid (0 to 15 years)
time_grid = np.linspace(0, max_time, 50)

# Compute percentiles across all patients
percentiles = compute_percentiles(final_data, time_grid)

# Display the plots immediately
st.markdown("## Biomarker Simulations")
plot_biomarker_trajectories(final_data, percentiles, time_grid)














# Run simulation button
if st.sidebar.button('Run Simulation'):
    # Convert parameters to the appropriate format
    params1 = [r_C1, r_H1, r_W1, r_A1, r_I1,
               alpha_HI1, alpha_AW1]
    st.session_state['params1'] = params1

    # Adjust noise level
    noise_C_std = noise_level * 40
    noise_H_std = noise_level * 2.5  # std_H
    noise_W_std = noise_level * 2.3  # std_W
    noise_A_std = noise_level * 0.9  # std_A
    noise_I_std = noise_level * 21   # std_I
    noise_std = [noise_C_std, noise_H_std, noise_W_std, noise_A_std, noise_I_std]

    # Generate initial conditions based on the input means
    num_patient = 3
    simu_init_inter = generate_initial_conditions(
        num_patient,
        mean_C_simu_inter,
        std_C_inter,
        mean_H_simu_inter,
        mean_W_simu_inter,
        mean_A_simu_inter,
        mean_I_simu_inter
    )
    
    # Time points
    num_simu_patient = num_patient
    time_points_simu = [[i for i in range(16)] for _ in range(num_simu_patient)]
    
    # Maxes
    K_C = 200.0
    K_H = 15.0
    K_W = 25.0
    K_A = 5.0
    K_I = 160.0
    maxes = K_C, K_H, K_W, K_A, K_I

    # Run the simulation
    final_data = concatenate_data_diff_noise(
        simu_init_inter, time_points_simu,
        maxes, params1, noise_std, noise=add_noise, type='hypo'
    )

    # Check for zero values in final_data
    contains_zero = any((patient_data[:, :-1] <= 0).any() for patient_data in final_data)
    if contains_zero:
        st.warning("The generated data contains zero or negative values. Please review your parameters or initial conditions.")
    else:
        st.success("No zero or negative values generated")


    
    # Visualize the results
    st.markdown("## Simulation Results")
    plot_simulation(final_data)
else:
    st.markdown("## Awaiting Simulation")
    st.markdown("Adjust parameters and click **Run Simulation** to see results.")


# Collect information from the UI elements and session_state
add_variability_state = 'T' if add_noise else 'F'
variability_level_value = noise_level if add_noise else "N/A"

# Collect mean values at birth for intermediate and severe groups
birth_means_var = [
    mean_H_simu_inter, mean_W_simu_inter, mean_A_simu_inter, mean_I_simu_inter, mean_C_simu_inter, std_C_inter
]

# Save parameters section
st.sidebar.markdown("### Save Your Parameters")
user_name = st.sidebar.text_input("Your Name")
user_comment = st.sidebar.text_area("Comments (Why did you choose these parameters?)")

# Ensure params1 exists in session state before accessing
if st.sidebar.button('Save Parameters'):
    if not user_name:
        st.warning("Please enter your name before saving.")
    elif 'params1' not in st.session_state:
        st.warning("Please run the simulation first to generate parameters.")
    else:
        save_parameters(
            user_name, 
            user_comment, 
            [float(p) for p in st.session_state['params1']],
            add_variability_state, 
            variability_level_value,
            birth_means_var
        )
        st.session_state['parameters_saved'] = True  # Set flag to True after saving

# Display the success message only once
# if st.session_state['parameters_saved']:
    # st.success("Parameters saved successfully!")
    


