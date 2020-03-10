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
	if data.output{i} == "TRUE" & data.predict{i} == "TRUE"
		data1 = [data1 data.(feature)(i)];
	end
	if data.output{i} == "TRUE" & data.predict{i} == "FALSE"
		data2 = [data2 data.(feature)(i)];
	end
end

for i = 1:s
	if data.output{i} == "FALSE" & data.predict{i} == "TRUE"
		data3 = [data3 data.(feature)(i)];
	end
	if data.output{i} == "FALSE" & data.predict{i} == "FALSE"
		data4 = [data4 data.(feature)(i)];
	end
end

% DISTRIBUTION FOR NON-BASE FEATURES

edges = [0 0.26 0.51 0.76 1.0];
% data1 = red
h1 = histogram(data1, edges, 'Normalization', 'probability'); 
h1.FaceColor = [1 0 0];
hold on
% data2 = blue
h1 = histogram(data2, edges, 'Normalization', 'probability');
h1.FaceColor = [0 0 1];
grid on;

t = char(feature);
t(strfind(feature, "_")) = ' ';
feature = string(t);

title("Distribution of " + feature, 'FontSize', 18);
xlabel(feature, 'FontSize', 14);
ylabel('Occurance', 'FontSize', 14);
legend('Predicted True/True','Predicted False/True')
xticks([.13 .375 .635 .88])
set(gca,'xticklabel',{'A','T','C', 'G'});
hold off
saveas(h1, "../features/true_" + feature + ".png");
hold off

edges = [0 0.26 0.51 0.76 1.0];
% data1 = red
h2 = histogram(data3, edges, 'Normalization', 'probability'); 
h2.FaceColor = [1 0 0];
hold on
% data2 = blue
h2 = histogram(data4, edges, 'Normalization', 'probability'); 
h2.FaceColor = [0 0 1];
grid on;

t = char(feature);
t(strfind(feature, "_")) = ' ';
feature = string(t);

title("Distribution of " + feature, 'FontSize', 18);
xlabel(feature, 'FontSize', 14);
ylabel('Occurance', 'FontSize', 14);
legend('Predicted False/True','Predicted False/False')
xticks([.13 .375 .635 .88])
set(gca,'xticklabel',{'A','T','C', 'G'});
hold off
saveas(h2, "../features/false_" + feature + ".png");
