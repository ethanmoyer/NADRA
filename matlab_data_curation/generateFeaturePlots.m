% This function accepts the location of a mismatched table (.csv) prefixed
% with 'mismatched_' and calls data1DPlot based on the feature set of the
% table, generating mismatched distributions for each feature.

function generateFeaturePlots (location)
data = readtable(location);
features = data.Properties.VariableNames;
% DISPLAY DISTRIBUTION ACROSS S AND NOT JUST BASES
s = size(features, 2) - 2;
for i = 2:47
	features{i};
	data1DPlot(location, features{i})
end
end