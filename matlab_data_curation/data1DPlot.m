% This function accepts the location of mismatched tables (.csv) prefixed
% with 'mismatched_' and a feature from the data set and plots the
% distrubition of the feature across both true mismatches and false
% mismatches. These plots are saved to the ../feature/ directory prefixed
% with the technique used, whether it's mapping true or false, and the
% feature.

function data1DPlot (location, feature)
data = readtable(location);
data = rmmissing(data);
s = height(data);

data1 = [];
data2 = [];

data3 = [];
data4 = [];

for i = 1:s
	if string(data.output{i}).lower == "true" & string(data.predicted{i}).lower == "true"
		data1 = [data1 data.(feature)(i)];
	end
	if string(data.output{i}).lower == "true" & string(data.predicted{i}).lower == "false"
		data2 = [data2 data.(feature)(i)];
	end
end

for i = 1:s
	if string(data.output{i}).lower == "false" & string(data.predicted{i}).lower == "true"
		data3 = [data3 data.(feature)(i)];
	end
	if string(data.output{i}).lower == "false" & string(data.predicted{i}).lower == "false"
		data4 = [data4 data.(feature)(i)];
	end
end

% DISTRIBUTION FOR NON-BASE FEATURES

% Isolates the name of the current feature being analyzed.
t = char(feature);
t(strfind(feature, "_")) = ' ';
feature = string(t);

% Isolates the name of the current test being analyzed.
t = char(location);
l = strfind(location, "_");
test = t(l(2) + 1: l(3) - 1);

% Sets the edges for the nucleotide distribution.
edges = [0 0.26 0.51 0.76 1.0];
% data1 = red
h1 = histogram(data1, edges, 'Normalization', 'probability'); 
h1.FaceColor = [1 0 0];
hold on
% data2 = blue
h1 = histogram(data2, edges, 'Normalization', 'probability');
h1.FaceColor = [0 0 1];
grid on;

title("Distribution of " + feature, 'FontSize', 18);
xlabel(feature, 'FontSize', 14);
ylabel('Occurance', 'FontSize', 14);
legend('Predicted True/True','Predicted False/True')
xticks([.13 .375 .635 .88])
set(gca,'xticklabel',{'A','T','C', 'G'});
hold off

saveas(h1, "../features/" + test + "_true_" + feature + ".png");
hold off

% data3 = red
h2 = histogram(data3, edges, 'Normalization', 'probability'); 
h2.FaceColor = [1 0 0];
hold on
% data4 = blue
h2 = histogram(data4, edges, 'Normalization', 'probability'); 
h2.FaceColor = [0 0 1];
grid on;

title("Distribution of " + feature, 'FontSize', 18);
xlabel(feature, 'FontSize', 14);
ylabel('Occurance', 'FontSize', 14);
legend('Predicted False/True','Predicted False/False')
xticks([.13 .375 .635 .88])
set(gca,'xticklabel',{'A','T','C', 'G'});
hold off

% Saves feature plot in feature directory.
saveas(h2, "../features/" + test + "_false_" + feature + ".png");
