function preprocessAngFile(angFile, outputFolder)

  fprintf("\n\nAccessing file: %s\n", angFile)

  IPF_Direction = vector3d.Y;
  % AlphaEbsdFolder = 'TestAlphaInput/';
  % BetaEbsdFolder = 'FullResolutionBeta/';
  BetaThreshold = 2.5;
  MinBetaSizePixels = 15;



%LocalEbsdBaseName = erase(FileList(iFile).name,'.ang');
AlphaEBSD= EBSD.load(angFile);


spacing = AlphaEBSD.unitCell(1)*2;


alphaName='Titanium (Alpha)';
betaName = 'Titanium (Beta)';


if ShouldPlotReconstructionSteps
  H = figure; plot(AlphaEBSD(alphaName),AlphaEBSD(alphaName).orientations,'figSize','large');
  saveas(H, strcat(AlphaEbsdFolder,LocalEbsdBaseName,'.png'));
end

beta2alpha = orientation.Burgers(AlphaEBSD(betaName).CS,AlphaEBSD(alphaName).CS);
round2Miller(beta2alpha)
%%% that alligns (110) plane of the beta phase with the (0001) plane of the alpha phase and the [1-11] direction of the beta phase with the [2110] direction of the alpha phase
% reconstruct grains
[grains,AlphaEBSD.grainId] = calcGrains(AlphaEBSD('indexed'),'threshold',1.5*degree,...
  'removeQuadruplePoints');
grains = smooth(grains,1,'moveTriplePoints');
% plot all alpha pixels
if ShouldPlotReconstructionSteps
J = figure; plot(AlphaEBSD(alphaName),AlphaEBSD(alphaName).orientations,...
  'figSize','large');
end
% and on top the grain boundaries
if ShouldPlotReconstructionSteps
  hold on
  plot(grains.boundary,'linewidth',2);
  hold off
  saveas(J, strcat(AlphaEbsdFolder,LocalEbsdBaseName,'Marked','.png'));
end
close all % this removes the images 


%%% Next we extract all alpha - alpha - alpha triple junctions and use the command calcParent to find for each of these triple junctions the best fitting parent orientations.

% extract all alpha - alpha - alpha triple points
tP = grains.triplePoints(alphaName,alphaName,alphaName);

% compute for each triple point the best fitting parentId and how well the fit is
tPori = grains(tP.grainId).meanOrientation;
[parentId, fit] = calcParent(tPori,beta2alpha,'numFit',2,'id','threshold',5*degree);


%%% The command calcParent returns for each child orientation a parentId which
% allows us later to compute the parent orientation from the child orientation.
% Furthermore, the command return for each triple junction the misfit between 
% the adjecent parent orientations in radiant.
% Finally, the option 'numFit',2 causes calcParent to return not only the best fit but also the second best fit
consistenTP = fit(:,1) < 2.5*degree & fit(:,2) > 2.5*degree;

%%%%% Recover beta grains from consistent triple junctions

% get a unique parentId vote for each grain
[parentId, numVotes] = majorityVote( tP(consistenTP).grainId, ...
  parentId(consistenTP,:,1), max(grains.id),'strict');
% lets store the parent grains into a new variable
parentGrains = grains;
% change orientations of consistent grains from child to parent
parentGrains(numVotes>2).meanOrientation = ...
  variants(beta2alpha,grains(numVotes>2).meanOrientation,parentId(numVotes>2));
% update all grain properties that are related to the mean orientation
parentGrains = parentGrains.update;

% define a color key
ipfKey = ipfColorKey(AlphaEBSD(betaName));
ipfKey.inversePoleFigureDirection = IPF_Direction; 

% and plot
if ShouldPlotReconstructionSteps
  K = figure;
  plot(parentGrains(betaName),ipfKey.orientation2color(parentGrains(betaName).meanOrientation),'figSize','large')
  PlotPath = strcat(ReconstructionPlotFolder,LocalEbsdBaseName,'Step1','.png');
  saveas(K, PlotPath);
  close gcf;
end


%{
We observe that this first step already results in very many Beta grains. However, the grain boundaries are still the boundaries of the original alpha grains. To overcome this, we merge all Beta grains that have a misorientation angle smaller then 2.5 degree.

As an additional consistency check we verify that each parent grain has been reconstructed from at least 2 child grains. To this end we first make a testrun the merge operation and then revert all parent grains that that have less then two childs. This step may not nessesary in many case.
%}



% test run of the merge operation
[~,parentId] = merge(parentGrains,'threshold',BetaThreshold*degree,'testRun');

% count the number of neighbouring child that would get merged with each child
counts = accumarray(parentId,1);

% revert all beta grains back to alpha grains if they would get merged with
% less then 1 other child grains
setBack = counts(parentId) < 2 & grains.phaseId == grains.name2id(alphaName);
parentGrains(setBack).meanOrientation = grains(setBack).meanOrientation;
parentGrains = parentGrains.update;



% merge beta grains
[parentGrains,parentId] = merge(parentGrains,'threshold',2.5*degree);

% set up a EBSD map for the parent phase
parentEBSD = AlphaEBSD;

% and store there the grainIds of the parent grains
parentEBSD('indexed').grainId = parentId(AlphaEBSD('indexed').grainId);

% figure;plot(parentGrains(betaName), ...
% ipfKey.orientation2color(parentGrains(betaName).meanOrientation),'figSize','large')
% saveas('Step2_BetaReconstructedGrainsMTEX.png')
if ShouldPlotReconstructionSteps
    L = figure;
    plot(parentGrains(betaName),ipfKey.orientation2color(parentGrains(betaName).meanOrientation),'figSize','large')
    PlotPath = strcat(ReconstructionPlotFolder,LocalEbsdBaseName,'Step2','.png');
    saveas(L, PlotPath);
    close gcf;
end




% all neighbouring alpha - beta grains
grainPairs = neighbors(parentGrains(alphaName), parentGrains(betaName));


% extract the corresponding meanorientations
oriAlpha = parentGrains( grainPairs(:,1) ).meanOrientation;
oriBeta = parentGrains( grainPairs(:,2) ).meanOrientation;

% compute for each alpha / beta pair of grains the best fitting parentId
[parentId, fit] = calcParent(oriAlpha,oriBeta,beta2alpha,'numFit',2,'id');

%{
Similarly, as in the first step the command calcParent returns a list of parentId that allows
the convert the child orientations into parent orientations using the command variants and the
fitting to the given parent orientation. Similarly, as for the triple point we select only 
those alpha beta pairs such that the fit is below the threshold of 2.5 degree and at the same
time the second best fit is above 2.5 degree.
%}

% consistent pairs are those with a very small misfit
consistenPairs = fit(:,1) < 5*degree & fit(:,2) > 5*degree;


parentId = majorityVote( grainPairs(consistenPairs,1), ...
  parentId(consistenPairs,1), max(parentGrains.id));

% change grains from child to parent
hasVote = ~isnan(parentId);
parentGrains(hasVote).meanOrientation = ...
  variants(beta2alpha, parentGrains(hasVote).meanOrientation, parentId(hasVote));

% update grain boundaries
parentGrains = parentGrains.update;

% merge new beta grains into the old beta grains
[parentGrains,parentId] = merge(parentGrains,'threshold',5*degree);

% update grainId in the ebsd map
parentEBSD('indexed').grainId = parentId(parentEBSD('indexed').grainId);

% plot the result
color = ipfKey.orientation2color(parentGrains(betaName).meanOrientation);

if ShouldPlotReconstructionSteps
  M = figure;
  plot(parentGrains(betaName),color,'linewidth',2)
  PlotPath = strcat(ReconstructionPlotFolder,LocalEbsdBaseName,'Step3','.png');
  saveas(M, PlotPath);
  close gcf;
end


%%%% Reconstruct beta orientations in EBSD map

% consider only original alpha pixels that now belong to beta grains

isNowBeta = parentGrains.phaseId(max(1,parentEBSD.grainId)) == AlphaEBSD.name2id(betaName) &...
  parentEBSD.phaseId == AlphaEBSD.name2id(alphaName);


%%% Next we can use once again the function calcParent to recover the original
% beta orientation from the measured alpha orientation giving the mean beta orientation of the grain.

% update beta orientation
[parentEBSD(isNowBeta).orientations, fit] = calcParent(parentEBSD(isNowBeta).orientations,...
  parentGrains(parentEBSD(isNowBeta).grainId).meanOrientation,beta2alpha);


%We obtain even a measure fit for the corespondence between the beta orientation reconstructed
% for a single pixel and the beta orientation of the grain. Lets visualize this measure of fit



% the beta phase
if ShouldPlotReconstructionSteps
  N = figure; 
  plot(parentEBSD(isNowBeta),fit ./ degree,'figSize','large');
  mtexColorbar
  setColorRange([0,5])
  mtexColorMap('LaboTeX')

  hold on
  plot(parentGrains.boundary,'lineWidth',2)
  hold off

  PlotPath = strcat(ReconstructionPlotFolder,LocalEbsdBaseName,'Step4','.png');
  saveas(N, PlotPath);
  close gcf;
end



% Lets finaly plot the reconstructed beta phase
if ShouldPlotReconstructionSteps
  O = figure; 
  plot(parentEBSD(betaName),ipfKey.orientation2color(parentEBSD(betaName).orientations),'figSize','large')
  PlotPath = strcat(ReconstructionPlotFolder,LocalEbsdBaseName,'Step5','.png');
  saveas(O, PlotPath);
end




%%%%%%%%%% Denoising of the reconstructed beta phase %%%%%%%%%%

[parentGrains,parentEBSD.grainId] = calcGrains(parentEBSD('indexed'),'angle',5*degree);

% remove all the small grains
parentEBSD = parentEBSD(parentGrains(parentGrains.grainSize > MinBetaSizePixels ));

% redo grain reconstruction
[parentGrains,parentEBSD.grainId] = calcGrains(parentEBSD('indexed'),'angle',5*degree);

% smooth the grains a bit
parentGrains = smooth(parentGrains,5);


% Finally, we denoise the remaining beta orientations and at the same time
% fill the empty holes. We choose a very small smoothing parameter alpha
% to keep as many details as possible.

F= halfQuadraticFilter;
F.alpha = 0.1;
parentEBSD = smooth(parentEBSD,F,'fill',parentGrains);

% plot the resulting beta phase
if ShouldPlotReconstructionSteps
  P = figure;
  plot(parentEBSD(betaName),ipfKey.orientation2color(parentEBSD(betaName).orientations),'figSize','large');
  hold on
  plot(parentGrains.boundary,'lineWidth',3)
  hold off
  PlotPath = strcat(ReconstructionPlotFolder,LocalEbsdBaseName,'Step6','.png');
  saveas(P, PlotPath);
end
    
if ShouldPlotFinalReconstructionOnly 
  Q = figure;
  plot(parentEBSD(betaName),ipfKey.orientation2color(parentEBSD(betaName).orientations),'figSize','large');
  hold on
  plot(parentGrains.boundary,'lineWidth',3)
  hold off
  PlotPath = strcat(ReconstructionPlotFolder,LocalEbsdBaseName,'Step6','.png');
  saveas(Q, PlotPath);
end


% fprintf(' the final Beta reconstructed ang file can be imported from "parentEBSD" \n');
% fprintf(" using <ebsd variable name>.export(' <outputfilename>.ang')\n");




gridParentEBSD = parentEBSD.gridify;% great 
gridParentEBSD_Rotations = gridParentEBSD.rotations;
gridParentEBSD_Phase = gridParentEBSD.phase;
sizeGrid = size(gridParentEBSD);
nColsX = sizeGrid(1);
nColsY = sizeGrid(2);
numelRotations = numel(sizeGrid);
% now to make the linearized Euler Angles: 

linearizedEulerAngles = zeros(numelRotations,3);
linearizedPhase = zeros(numelRotations,1);
counter = 0;
for ii = 1: nColsX
  for jj = 1:nColsY
    counter = counter+1;
    linearizedEulerAngles(counter,1) = gridParentEBSD_Rotations(ii,jj).phi1;
    linearizedEulerAngles(counter,2) = gridParentEBSD_Rotations(ii,jj).Phi;
    linearizedEulerAngles(counter,3) = gridParentEBSD_Rotations(ii,jj).phi2;
    
    if isnan(linearizedEulerAngles(counter,1))
      linearizedEulerAngles(counter,:) = 0;  
    end
    
    linearizedPhase(counter) = gridParentEBSD_Phase(ii,jj); % the -1 is important.
  end
end
OutFileName = strcat(BetaEbsdFolder,LocalEbsdBaseName,'.ang');
HeaderFileName = 'EBSD_Header.txt';
% now we need to make the linerized features. 


ExportAngFile(nColsX,nColsY,linearizedEulerAngles,linearizedPhase,OutFileName,HeaderFileName,spacing);
% OutFileName = [BetaEbsdFolder,LocalEbsdBaseName,'Zest.ang'];
% parentEBSD.export(OutFileName)
% pause


end



function  ExportAngFile(nColsX,nColsY,linearizedEulerAngles,linearizedPhase,OutFileName,HeaderFileName,spacing)
  FID_Out = fopen(OutFileName,'w');


  fidIn = fopen(HeaderFileName,'r');
  tline = fgetl(fidIn);
  encoutneredSize = false;
  while ischar(tline)
      tline = fgetl(fidIn);
      % now we need to parse
      
      if encoutneredSize == false
          if contains(tline,'XSTEP')
  %             we want to rewrite the following lines
  %             with updated values. So look for 'XSTEP'
  %             and rewrite the 5 lines u wan't
  %             # XSTEP: 1
  %             # YSTEP: 1
  %             # NCOLS_ODD: 501
  %             # NCOLS_EVEN: 501
  %             # NROWS: 501


              % write the thing
              encoutneredSize = true;
              % write the two lines I need
                          %phi1 Phi2 Phi3 X,Y ,idk,idk,FeatureId,
              %f %f %f %i %i %i %i %i %i %i
              WriteStr = strcat('# XSTEP: ',num2str(spacing));
              fprintf(FID_Out,'%s \n',WriteStr);
              WriteStr = strcat('# YSTEP: ',num2str(spacing));
              fprintf(FID_Out,'%s \n',WriteStr);
              WriteStr = strcat('# NCOLS_ODD  ',num2str(nColsY)); % not sure if this is correct, or need to switch X and Y
              fprintf(FID_Out,'%s \n',WriteStr);
              WriteStr =strcat('# NCOLS_EVEN  ',num2str(nColsY));
              fprintf(FID_Out,'%s \n',WriteStr);
              WriteStr =strcat('# NROWS  ',num2str(nColsX));
              fprintf(FID_Out,'%s \n',WriteStr);
              
              tline = fgetl(fidIn);% also grab the next line so I don't write it
              tline = fgetl(fidIn);% also grab the next line so I don't write it
              tline = fgetl(fidIn);% also grab the next line so I don't write it
              tline = fgetl(fidIn);% also grab the next line so I don't write it

          else
              fprintf(FID_Out,'%s \n',tline);
          end
      else
          if tline == -1
              %nothing 
          else
          fprintf(FID_Out,'%s \n',tline);
          end
      end
      
  end

  numelOutput = nColsX *nColsY;
  WriteMat = ones(numelOutput,10);
  WriteMat(:,1:3) = linearizedEulerAngles;
  WriteMat(:,8) = linearizedPhase;
  % now we need to do x and y
  % x first
  counter = 0;
  for ii = 1:nColsX
      for jj = 1: nColsY
          counter = counter+1;
          WriteMat(counter,5) = (ii-1)*spacing;
          WriteMat(counter,4) = (jj-1)*spacing;
          
      end
  end
  %WriteMat(end,8) = 2;
  fprintf(FID_Out,'%f %f %f %f %f %i %i %i %i %i \n',WriteMat'); % probably fine
  fclose(FID_Out);
  fclose(fidIn);

end