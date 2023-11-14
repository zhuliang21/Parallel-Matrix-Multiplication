library(tidyverse)

# Create a data frame with the results
results <- data.frame(
  N = c(rep(2^8, 3), rep(2^10, 3), rep(2^12, 3)),
  P = c(7, 49, 343, 7, 49, 343, 7, 49, 343),
  Time = c(0.029430, 0.010317, 0.040137, 
           1.757423, 0.284435, 0.274502, 
           63.811983, 9.049598, 2.902417)) %>%
  group_by(N) %>%
  # speedup is define by 7 * time(P=7) / time(P)
  mutate(Speedup = 7 * Time[1] / Time) %>%
  # drop observations with P = 7
  filter(P != 7) %>%
  # drop the time column
  ungroup()



# Plot the speedup curve
graph <- ggplot(results, aes(x = P, y = Speedup, group = N, color = as.factor(N))) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = seq(0, 343, by = 100), limits = c(0, 343)) +
  scale_y_continuous(breaks = seq(0, 343, by = 100), limits = c(0, 343)) +
  # also plot the dash diagonal line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  scale_color_discrete(name = "Matrix Size (N)", labels = c("2^8", "2^10", "2^12")) +
  labs(title = "Speedup Curve of Strassen Algorithm", x = "Number of Cores (P)", y = "Speedup") +
  theme_minimal()

# Save the plot
ggsave("~/project4/report/speedup_curve.png", graph, width = 4, height = 3, dpi = 300)

