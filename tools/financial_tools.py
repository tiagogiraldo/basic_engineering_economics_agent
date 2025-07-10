from typing import Optional
from engineconomics import time_value
from langchain.tools import tool

tv = time_value()

# Define your tool function
@tool
def time_value_tool(CF: float, F: str, i: float, n: float, g: Optional[float] = None) -> float:
    """
    Computes time value of money factors using standard financial formulas.

    Parameters:
    CF (float): Assessed cash flow
    F (str): Factor type (e.g., "P/F", "F/A", "P/G", "P/g")
    i (float): Effective interest rate
    n (float): Term in periods
    g (float): Geometric gradient (optional)

    Returns:
    dictionary
    """
    output = tv.cfv(CF, F, i, n, g)

    return output