library(ggplot2)

# Create a data frame with the results
results <- data.frame(
  N = c(rep(2^8, 3), rep(2^10, 3), rep(2^12, 3)),
  P = c(2^2, 2^4, 2^6, 2^2, 2^4, 2^6, 2^2, 2^4, 2^6),
  Speedup = c(0.090545/0.020182, 0.090545/0.006294, 0.090545/0.003478,
              6.119090/1.163175, 6.119090/0.310421, 6.119090/0.095925,
              707.989130/73.364430, 707.989130/18.900791, 707.989130/5.574106)
)

# Create a function to transform the legend labels
label_transform <- function(n) {
  sapply(n, function(x) paste0("2^", log2(as.numeric(x))))
}

# Plot the speedup curve
graph <- ggplot(results, aes(x = P, y = Speedup, group = N, color = as.factor(N))) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = seq(0, max(results$Speedup), by = 16)) +
  scale_y_continuous(breaks = seq(0, max(results$Speedup), by = 16)) +
  scale_color_discrete(labels = label_transform) +
  labs(title = "Speedup Curve", x = "Number of Processors (P)", y = "Speedup", color = "Matrix Size (N)") +
  theme_minimal()

# Save the plot
ggsave("~/project3/report/speedup_curve.png", graph, width = 4, height = 5, dpi = 300)

library(ggplot2)

# Create a data frame with the new results
results <- data.frame(
  N = c(rep(2^8, 5), rep(2^10, 5), rep(2^12, 5)),
  P = c(2^2, 2^3, 2^4, 2^5, 2^6, 2^2, 2^3, 2^4, 2^5, 2^6, 2^2, 2^3, 2^4, 2^5, 2^6),
  Time = c(0.039433, 0.020771, 0.011038, 0.009600, 0.006572,
           2.123300, 1.203297, 0.635084, 0.353849, 0.175946,
           73.409207, 37.168124, 20.105683, 10.941110, 6.431338)
)

# Calculate the speedup using the new definition
results$BaseTime <- with(results, ave(Time, N, FUN = function(x) x[1]))
results$Speedup <- with(results, BaseTime / Time * 4)

# Create a function to transform the legend labels
label_transform <- function(n) {
  sapply(n, function(x) paste0("2^", log2(as.numeric(x))))
}

# Plot the speedup curve
graph <- ggplot(results, aes(x = P, y = Speedup, group = N, color = as.factor(N))) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = seq(0, 64, by = 16), limits=c(0, 64)) +
  scale_y_continuous(breaks = seq(0, 64, by = 16), limits=c(0, 64)) +
  scale_color_discrete(labels = label_transform) +
  labs(title = "Speedup Curve", x = "Number of Processors (P)", y = "Speedup", color = "Matrix Size (N)") +
  theme_minimal()

# Save the plot
ggsave("~/project3/report/speedup_curve_appendex.png", graph, width = 5, height = 4, dpi = 300)
