document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('matchForm');
    const resumeInput = document.getElementById('resume');
    const jdFileInput = document.getElementById('jdFile');
    const jdTextInput = document.getElementById('jdText');
    const submitButton = document.getElementById('submitButton');
    const loadingElement = document.getElementById('loading');
    const errorElement = document.getElementById('error');
    const resultsSection = document.getElementById('results');

    // Toggle between file upload and text input for job description
    document.getElementById('jdInputType').addEventListener('change', (e) => {
        const isFile = e.target.value === 'file';
        jdFileInput.parentElement.style.display = isFile ? 'block' : 'none';
        jdTextInput.parentElement.style.display = isFile ? 'none' : 'block';
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Reset UI
        errorElement.classList.remove('active');
        resultsSection.classList.remove('active');
        submitButton.disabled = true;
        loadingElement.classList.add('active');

        try {
            const formData = new FormData();
            // Append all selected resumes
            for (let i = 0; i < resumeInput.files.length; i++) {
                formData.append('resumes', resumeInput.files[i]);
            }

            // Handle job description input
            if (jdFileInput.files[0]) {
                formData.append('job_description', jdFileInput.files[0]);
            } else if (jdTextInput.value.trim()) {
                formData.append('jd_text', jdTextInput.value.trim());
            } else {
                throw new Error('Please provide either a job description file or text');
            }

            const response = await fetch(`${config.API_URL}/api/match`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'An error occurred while processing your request');
            }

            const results = await response.json();
            displayResults(results);

        } catch (error) {
            errorElement.textContent = error.message;
            errorElement.classList.add('active');
        } finally {
            submitButton.disabled = false;
            loadingElement.classList.remove('active');
        }
    });

    let globalResultsArray = [];

    function displayResults(results) {
        // If multiple results, show a summary for each
        if (results.results && Array.isArray(results.results)) {
            let summaryHtml = '';
            let firstValidResult = null;
            globalResultsArray = results.results;
            // Populate dropdown
            const resumeSelect = document.getElementById('resumeSelect');
            if (resumeSelect) {
                resumeSelect.innerHTML = '';
                results.results.forEach((res, idx) => {
                    let label = res.resume && (res.resume.name || res.resume.email) ? `${res.resume.name || res.resume.email}` : `Resume ${idx + 1}`;
                    if (res.error) label = `Error: ${res.filename || 'Resume ' + (idx + 1)}`;
                    const option = document.createElement('option');
                    option.value = idx;
                    option.textContent = label;
                    resumeSelect.appendChild(option);
                });
                resumeSelect.onchange = function() {
                    const idx = parseInt(this.value);
                    if (!isNaN(idx) && globalResultsArray[idx] && !globalResultsArray[idx].error) {
                        displaySingleResult(globalResultsArray[idx]);
                    } else {
                        document.getElementById('resumeDetails').style.display = 'none';
                        document.getElementById('jdDetails').style.display = 'none';
                    }
                };
            }
            results.results.forEach((res, idx) => {
                if (res.error) {
                    summaryHtml += `<div class="score-card error"><h3>Resume ${idx + 1}: Error</h3><div style="color:red;">${res.filename ? res.filename + ': ' : ''}${res.error}</div></div>`;
                } else {
                    const matchScore = res.match_score;
                    summaryHtml += `<div class="score-card"><h3>Resume ${idx + 1}: ${res.resume.name || res.resume.email || 'N/A'}</h3>`;
                    summaryHtml += `<div>Overall Score: <strong>${(matchScore.overall_score * 100).toFixed(1)}%</strong></div>`;
                    summaryHtml += `<div>Skill Score: ${(matchScore.skill_score * 100).toFixed(1)}%</div>`;
                    summaryHtml += `<div>Experience Score: ${(matchScore.experience_score * 100).toFixed(1)}%</div>`;
                    summaryHtml += `<div>Education Score: ${(matchScore.education_score * 100).toFixed(1)}%</div>`;
                    summaryHtml += `<div>Semantic Score: ${(matchScore.semantic_score * 100).toFixed(1)}%</div>`;
                    summaryHtml += `<div>Matched Skills: ${matchScore.skill_matches.matched.join(', ')}</div>`;
                    summaryHtml += `</div>`;
                    if (!firstValidResult) firstValidResult = res;
                }
            });
            resultsSection.innerHTML = `<h2>Match Results</h2>${summaryHtml}`;
            resultsSection.classList.add('active');
            // Show details for the first valid result
            if (firstValidResult) {
                if (resumeSelect) resumeSelect.value = results.results.findIndex(r => r === firstValidResult);
                displaySingleResult(firstValidResult);
            } else {
                // Hide details if all errored
                document.getElementById('resumeDetails').style.display = 'none';
                document.getElementById('jdDetails').style.display = 'none';
            }
        } else {
            // Single result fallback
            globalResultsArray = [results];
            const resumeSelect = document.getElementById('resumeSelect');
            if (resumeSelect) {
                resumeSelect.innerHTML = '';
                const option = document.createElement('option');
                option.value = 0;
                option.textContent = results.resume && (results.resume.name || results.resume.email) ? `${results.resume.name || results.resume.email}` : `Resume 1`;
                resumeSelect.appendChild(option);
                resumeSelect.onchange = null;
            }
            displaySingleResult(results);
        }
    }

    function displaySingleResult(results) {
        const matchScore = results.match_score;
        // Update overall score
        const overallScoreElem = document.getElementById('overallScore');
        if (overallScoreElem) overallScoreElem.textContent = (matchScore.overall_score * 100).toFixed(1) + '%';

        // Update skill matches
        const skillMatches = matchScore.skill_matches;
        const matchedSkillsElem = document.getElementById('matchedSkills');
        if (matchedSkillsElem) matchedSkillsElem.innerHTML = skillMatches.matched.map(skill => `<li>${skill}</li>`).join('');
        const missingSkillsElem = document.getElementById('missingSkills');
        if (missingSkillsElem) missingSkillsElem.innerHTML = skillMatches.missing.map(skill => `<li>${skill}</li>`).join('');
        const semanticSkillsElem = document.getElementById('semanticSkills');
        if (semanticSkillsElem) semanticSkillsElem.innerHTML = skillMatches.semantically_matched.map(([jd, resume, score]) => `<li>${jd} → ${resume} (${(score * 100).toFixed(1)}%)</li>`).join('');

        // Update education matches
        const educationMatches = matchScore.education_matches;
        const matchedEducationElem = document.getElementById('matchedEducation');
        if (matchedEducationElem) matchedEducationElem.innerHTML = educationMatches.matched.map(edu => `<li>${edu}</li>`).join('');
        const missingEducationElem = document.getElementById('missingEducation');
        if (missingEducationElem) missingEducationElem.innerHTML = educationMatches.missing.map(edu => `<li>${edu}</li>`).join('');

        // Update experience matches
        const experienceMatches = matchScore.experience_matches;
        const matchedExperienceElem = document.getElementById('matchedExperience');
        if (matchedExperienceElem) matchedExperienceElem.innerHTML = experienceMatches.matched.map(exp => `<li>${exp}</li>`).join('');
        const missingExperienceElem = document.getElementById('missingExperience');
        if (missingExperienceElem) missingExperienceElem.innerHTML = experienceMatches.missing.map(exp => `<li>${exp}</li>`).join('');

        // Update detailed scores
        if (matchScore.calculated_experience_years) {
            const calculatedExperienceElem = document.getElementById('calculatedExperience');
            if (calculatedExperienceElem) calculatedExperienceElem.textContent = matchScore.calculated_experience_years.toFixed(1) + ' years';
        }
        if (matchScore.semantic_score) {
            const semanticScoreElem = document.getElementById('semanticScore');
            if (semanticScoreElem) semanticScoreElem.textContent = (matchScore.semantic_score * 100).toFixed(1) + '%';
        }
        if (matchScore.skill_score) {
            const skillScoreElem = document.getElementById('skillScore');
            if (skillScoreElem) skillScoreElem.textContent = (matchScore.skill_score * 100).toFixed(1) + '%';
        }
        if (matchScore.experience_score) {
            const experienceScoreElem = document.getElementById('experienceScore');
            if (experienceScoreElem) experienceScoreElem.textContent = (matchScore.experience_score * 100).toFixed(1) + '%';
        }
        if (matchScore.education_score) {
            const educationScoreElem = document.getElementById('educationScore');
            if (educationScoreElem) educationScoreElem.textContent = (matchScore.education_score * 100).toFixed(1) + '%';
        }

        // Show results section
        resultsSection.classList.add('active');

        // Populate Resume Details
        const resumeDetails = results.resume;
        const resumeNameElem = document.getElementById('resumeName');
        if (resumeNameElem) resumeNameElem.textContent = resumeDetails.name || 'N/A';
        const resumeEmailElem = document.getElementById('resumeEmail');
        if (resumeEmailElem) resumeEmailElem.textContent = resumeDetails.email || 'N/A';
        const resumePhoneElem = document.getElementById('resumePhone');
        if (resumePhoneElem) resumePhoneElem.textContent = resumeDetails.phone || 'N/A';
        const resumeLocationElem = document.getElementById('resumeLocation');
        if (resumeLocationElem) resumeLocationElem.textContent = resumeDetails.location || 'N/A';
        const resumeSummaryElem = document.getElementById('resumeSummary');
        if (resumeSummaryElem) resumeSummaryElem.textContent = resumeDetails.summary || 'N/A';

        const resumeSkillsList = document.getElementById('resumeSkills');
        if (resumeSkillsList) resumeSkillsList.innerHTML = resumeDetails.skills.map(skill => `<li>${skill.name} (${skill.level})</li>`).join('');

        const resumeWorkList = document.getElementById('resumeWork');
        if (resumeWorkList) resumeWorkList.innerHTML = resumeDetails.work.map(job => `<li><strong>${job.position}</strong> at ${job.company} (${job.startDate} - ${job.endDate})<br/>${job.summary}</li>`).join('');

        const resumeEducationList = document.getElementById('resumeEducation');
        if (resumeEducationList) resumeEducationList.innerHTML = resumeDetails.education.map(edu => `<li><strong>${edu.studyType}</strong> from ${edu.institution} (${edu.startDate} - ${edu.endDate})</li>`).join('');

        const resumeDetailsSection = document.getElementById('resumeDetails');
        if (resumeDetailsSection) {
            resumeDetailsSection.classList.add('active');
            resumeDetailsSection.style.display = 'block';
        }

        // Populate Job Description Details
        const jdDetails = results.job_description;
        const jdTitleElem = document.getElementById('jdTitle');
        if (jdTitleElem) jdTitleElem.textContent = jdDetails.title || 'N/A';
        const jdRequiredExperienceElem = document.getElementById('jdRequiredExperience');
        if (jdRequiredExperienceElem) jdRequiredExperienceElem.textContent = jdDetails.required_experience_years > 0 ? `${jdDetails.required_experience_years}+ years` : 'N/A';
        const jdRequiredEducationElem = document.getElementById('jdRequiredEducation');
        if (jdRequiredEducationElem) jdRequiredEducationElem.textContent = jdDetails.required_education || 'N/A';

        const jdRequiredSkillsList = document.getElementById('jdRequiredSkills');
        if (jdRequiredSkillsList) jdRequiredSkillsList.innerHTML = jdDetails.required_skills.map(skill => `<li>${skill}</li>`).join('');
        
        const jdDetailsSection = document.getElementById('jdDetails');
        if (jdDetailsSection) {
            jdDetailsSection.classList.add('active');
            jdDetailsSection.style.display = 'block';
        }
    }

    const feedbackForm = document.getElementById('feedbackForm');
    const feedbackTextInput = document.getElementById('feedbackText');
    const submitFeedbackBtn = document.getElementById('submitFeedbackBtn');
    const feedbackMessage = document.getElementById('feedbackMessage');

    feedbackForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        feedbackMessage.classList.remove('active');
        submitFeedbackBtn.disabled = true;

        try {
            const feedbackText = feedbackTextInput.value.trim();
            if (!feedbackText) {
                throw new Error('Please enter your feedback.');
            }

            const response = await fetch(`${config.API_URL}/api/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ feedback_text: feedbackText })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to submit feedback.');
            }

            feedbackMessage.textContent = 'Thank you for your feedback!';
            feedbackMessage.style.backgroundColor = '#dcfce7'; // Light green for success
            feedbackMessage.style.color = '#16a34a'; // Dark green for success
            feedbackMessage.classList.add('active');
            feedbackTextInput.value = ''; // Clear the textarea

        } catch (error) {
            feedbackMessage.textContent = error.message;
            feedbackMessage.style.backgroundColor = '#fee2e2'; // Light red for error
            feedbackMessage.style.color = '#ef4444'; // Dark red for error
            feedbackMessage.classList.add('active');
        } finally {
            submitFeedbackBtn.disabled = false;
        }
    });
}); 