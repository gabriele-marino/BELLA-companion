library(ape)
library(phangorn)
library(phytools)

tree1 <- read.nexus("outputs/summaries/eucovid/flights_over_populations/BELLA/mcc.nexus")
tree2 <- read.nexus("outputs/summaries/eucovid/flights_over_populations/GLM/mcc.nexus")

rf_norm <- RF.dist(tree1, tree2, normalize = TRUE)
cat("Normalized RF distance:", rf_norm, "\n")

tree1c <- unroot(tree1)
tree2c <- unroot(tree2)

common <- intersect(tree1c$tip.label, tree2c$tip.label)
tree1c <- drop.tip(tree1c, setdiff(tree1c$tip.label, common))
tree2c <- drop.tip(tree2c, setdiff(tree2c$tip.label, common))

cop <- cophylo(tree1c, tree2c, assoc = cbind(common, common))

pdf("figures/eucovid-cophylo.pdf", width = 20, height = 12)
plot(cop, fsize = 0.5, link.type = "curved", link.lwd = 1)
dev.off()
