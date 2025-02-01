import streamlit as st
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Example Black-Scholes functions
def black_scholes_call(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call_price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    put_price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Example Binomial model functions
def binomial_call(S, K, T, r, sigma, steps=100):
    dt = T / steps
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)
    # Initialize terminal values
    option_values = [max(0, S * (u ** j) * (d ** (steps - j)) - K) for j in range(steps + 1)]
    # Roll back through the tree
    for i in range(steps - 1, -1, -1):
        option_values = [exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j]) for j in range(i + 1)]
    return option_values[0]

def binomial_put(S, K, T, r, sigma, steps=100):
    dt = T / steps
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)
    # Initialize terminal values
    option_values = [max(0, K - S * (u ** j) * (d ** (steps - j))) for j in range(steps + 1)]
    # Roll back through the tree
    for i in range(steps - 1, -1, -1):
        option_values = [exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j]) for j in range(i + 1)]
    return option_values[0]

# Function to generate a heatmap data grid
def generate_heatmap_data(model, option_type, S_range, sigma_range, K, T, r, steps=100):
    prices = np.zeros((len(sigma_range), len(S_range)))
    for i, sigma_val in enumerate(sigma_range):
        for j, S_val in enumerate(S_range):
            if model == "Black-Scholes":
                if option_type == "Call":
                    prices[i, j] = black_scholes_call(S_val, K, T, r, sigma_val)
                else:
                    prices[i, j] = black_scholes_put(S_val, K, T, r, sigma_val)
            else:  # Binomial model
                if option_type == "Call":
                    prices[i, j] = binomial_call(S_val, K, T, r, sigma_val, steps)
                else:
                    prices[i, j] = binomial_put(S_val, K, T, r, sigma_val, steps)
    return prices

def main():
    st.title("Option Pricing Calculator")
    st.write("Select the pricing model, option type, and parameters from the sidebar.")

    # Sidebar inputs for model and option type
    st.sidebar.header("Pricing Model & Option Type")
    model = st.sidebar.selectbox("Choose pricing model", ["Black-Scholes", "Binomial"])
    option_type = st.sidebar.radio("Option Type", ["Call", "Put"])

    # Sidebar inputs for option parameters
    st.sidebar.header("Option Parameters")
    S = st.sidebar.number_input("Underlying Price (S)", min_value=0.0, value=100.0, step=1.0)
    K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=100.0, step=1.0)
    T = st.sidebar.number_input("Time to Maturity (T in years)", min_value=0.0, value=1.0, step=0.1)
    r = st.sidebar.number_input("Risk-free Rate (r, as a decimal)", min_value=0.0, value=0.05, step=0.01)
    sigma = st.sidebar.number_input("Volatility (σ, as a decimal)", min_value=0.0, value=0.2, step=0.01)

    # For the binomial model, allow the user to choose the number of steps
    if model == "Binomial":
        steps = st.sidebar.slider("Number of Steps", min_value=10, max_value=500, value=100, step=10)
    else:
        steps = 100  # default for consistency

    # Display current parameters
    st.subheader("Current Parameters")
    st.write(f"**Model:** {model}")
    st.write(f"**Option Type:** {option_type}")
    st.write(f"**Underlying Price (S):** {S}")
    st.write(f"**Strike Price (K):** {K}")
    st.write(f"**Time to Maturity (T):** {T} year(s)")
    st.write(f"**Risk-free Rate (r):** {r}")
    st.write(f"**Volatility (σ):** {sigma}")
    if model == "Binomial":
        st.write(f"**Number of Steps:** {steps}")

    # Calculate option price on button click
    if st.button("Calculate Option Price"):
        if model == "Black-Scholes":
            if option_type == "Call":
                price = black_scholes_call(S, K, T, r, sigma)
            else:
                price = black_scholes_put(S, K, T, r, sigma)
        else:  # Binomial model
            if option_type == "Call":
                price = binomial_call(S, K, T, r, sigma, steps)
            else:
                price = binomial_put(S, K, T, r, sigma, steps)
        st.success(f"Calculated {option_type} Option Price: {price:.2f}")

    # Additional section: Display Heatmap for Option Price sensitivity
    if st.sidebar.checkbox("Show Heatmap"):
        st.subheader("Heatmap of Option Prices")
        st.write("This heatmap shows how the option price changes as you vary the Underlying Price (S) and Volatility (σ).")

        # Define ranges for S and sigma (you can adjust these ranges)
        S_range = np.linspace(S * 0.8, S * 1.2, 50)       # 80% to 120% of current S
        sigma_range = np.linspace(sigma * 0.5, sigma * 1.5, 50)  # 50% to 150% of current sigma

        # Generate the grid of prices
        prices = generate_heatmap_data(model, option_type, S_range, sigma_range, K, T, r, steps)

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(prices, xticklabels=np.round(S_range, 1), yticklabels=np.round(sigma_range, 2),
                    ax=ax, cmap="viridis")
        ax.set_xlabel("Underlying Price (S)")
        ax.set_ylabel("Volatility (σ)")
        ax.set_title(f"{option_type} Option Price Heatmap ({model} Model)")
        st.pyplot(fig)

if __name__ == "__main__":
    main()



