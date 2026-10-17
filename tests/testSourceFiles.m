function tests = testSourceFiles
tests = functiontests(localfunctions);
end
function testRootFunctionsLoad(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
files = dir(fullfile(root, '*.m'));
for k = 1:numel(files)
    file = fullfile(root, files(k).name);
    source = fileread(file);
    % Scripts are experiment entry points, not safe to execute in unit tests.
    cleaned = regexprep(source, '(?m)^\s*%[^\n]*', '');
    if isempty(regexp(cleaned, '^\s*function\b', 'once'))
        continue;
    end
    [~, name] = fileparts(file);
    try
        nargin(name); % Compile/load each function without running experiments.
    catch exception
        fprintf('%s\n', evalc('checkcode(file)'));
        verifyFail(testCase, sprintf('%s: %s', files(k).name, exception.message));
    end
end
end
