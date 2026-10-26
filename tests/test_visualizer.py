import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualizer import plot_bar_chart, plot_line_chart, plot_pie_chart, plot_forecast, render_chart

# Test data
data = [
    {"country": "Germany", "orders": 122},
    {"country": "USA", "orders": 122},
    {"country": "Brazil", "orders": 83},
    {"country": "France", "orders": 77},
    {"country": "UK", "orders": 56},
]

# Test bar chart
buf = plot_bar_chart(data, "country", "orders", title="Orders per Country")
print("✅ plot_bar_chart OK, size:", len(buf.read()), "bytes")

# Test auto render
buf2 = render_chart(data, title="Auto Chart")
print("✅ render_chart OK, size:", len(buf2.read()), "bytes")

# Test pie chart
buf3 = plot_pie_chart(data, "country", "orders", title="Orders Distribution")
print("✅ plot_pie_chart OK, size:", len(buf3.read()), "bytes")

# Test forecast
forecast_data = {
    "model_used": "linear",
    "historical": [
        {"order_date": "1997-01-01", "total": 1000},
        {"order_date": "1997-02-01", "total": 1200},
        {"order_date": "1997-03-01", "total": 1100},
    ],
    "forecast": [1300, 1400, 1500],
}
buf4 = plot_forecast(forecast_data)
print("✅ plot_forecast OK, size:", len(buf4.read()), "bytes")

print("\n✅ visualizer.py works!")