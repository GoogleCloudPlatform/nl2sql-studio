# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

To get started contributing:

1. Sign a Contributor License Agreement (see details above).
2. Fork the repo, develop and test your code changes.
3. Run the linter locally. To run linter you need to 
   install flake8 and black,

   Use below commands to install the libraries
   ```
   pip install flask8
   ```

   ```
   pip install black
   ```

   The below command shows you the files that need 
   to be formatted

   ```
   black --check ./<your folder or file path>
   ```

   If above command shows any file that need to be formatted. Run below command to see more details on what lines in the file need to be fomratted

   ```
   flake8 ./<your folder or file path>
   ```

4. Develop using the following guidelines to help expedite your review:
    a. Ensure that your code adheres to the existing [style](https://google.github.io/styleguide).
    b. Ensure that your code has an appropriate set of unit tests which all pass.

5. Submit a pull request.


### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.