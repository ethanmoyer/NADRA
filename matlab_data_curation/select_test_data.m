% This function accepts the location of a table (.csv) with equally split 
% true and false outputs prefixed with "split_", the required number of
% samples from the table, and the proportion of true samples to include
% in new sample. The result is a curated table (.csv) prefixed with the
% proporiton of true samples and the number of samles.

function select_test_data(loc, n, p)
data = readtable(loc);
data = rmmissing(data);
l = size(data, 1);
% p is the proporiton of trues
data_true = data(randperm(floor(n * p)),:);
data_false = data(randperm(floor(n * (1 - p))) + l/2,:);
% l = size(data_true, 1);
table = [data_true; data_false];
l = size(table, 1);
table = table(randperm(l),:);

% Changes prefix of the file.
t = char(location);
t(1:strfind(location, "_")) = '';
loc = string(t);

s = "../data/" + p + "_" + n + "_" + loc;

% table = data(1:n, :);
writetable(table, s, 'Delimiter', ',')

end