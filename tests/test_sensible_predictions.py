"""Owner: D. The model reacts to inputs in the direction we expect."""


def _p_dropout(model, student, **changes):
    row = student.copy()
    for name, value in changes.items():
        row[name] = value
    classes = list(model.classes_)
    return model.predict_proba(row)[0][classes.index("Dropout")]


def _student(data_split):
    _, X_test, _, _ = data_split
    return X_test.iloc[[0]].copy()


def test_passing_more_courses_lowers_dropout_risk(trained_model, data_split):
    student = _student(data_split)
    approved = "Curricular units 1st sem (approved)"
    failing = _p_dropout(trained_model, student, **{approved: 0})
    passing = _p_dropout(trained_model, student, **{approved: 6})
    assert passing < failing


def test_higher_grades_lower_dropout_risk(trained_model, data_split):
    student = _student(data_split)
    grade = "Curricular units 1st sem (grade)"
    low = _p_dropout(trained_model, student, **{grade: 2})
    high = _p_dropout(trained_model, student, **{grade: 16})
    assert high < low
