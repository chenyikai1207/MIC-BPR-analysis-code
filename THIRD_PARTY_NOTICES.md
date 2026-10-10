# Dependencies and source attribution

This repository imports NumPy, SciPy, scikit-learn, threadpoolctl, joblib and,
for optional plotting, Matplotlib. Their packages and source trees are not
vendored here. Each dependency retains its own license and notices. Refer to
the installed package metadata for the applicable terms.

The exact LDA checks compare this project's computations with scikit-learn.
The private `_is_constant_feature` helper is imported from the pinned installed
package; no third-party library is copied and renamed as author-owned source.

The MAT-v5 parser, legacy epoch compatibility helper and scientific functions
were curated from this analysis project's producer code. They do not confer
rights to the source datasets or replace the providers' attribution requirements.

Source-study datasets and original-study code remain separately available from
the providers linked in the README. No original-study data, code archive or
third-party dependency source distribution is included here.
