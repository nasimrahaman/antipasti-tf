import Antipasti.utilities.pyutils2 as py2


def test_append_to_attribute():
    def _setup_object():
        class Foo(object):
            test_int = 3
            test_list = [4]
            test_tuple = (2,)
        object_ = Foo()
        return object_

    # Test append without duplicates
    obj_ = _setup_object()
    py2.append_to_attribute(obj_, 'test_int', 3, delist=True, prevent_duplicates=True)
    assert hasattr(obj_, 'test_int')
    assert obj_.test_int == 3

    obj_ = _setup_object()
    py2.append_to_attribute(obj_, 'test_int', [3], delist=True, prevent_duplicates=True)
    assert hasattr(obj_, 'test_int')
    assert obj_.test_int == [3, [3]]

    obj_ = _setup_object()
    py2.append_to_attribute(obj_, 'test_list', 4, delist=True, prevent_duplicates=True)
    assert hasattr(obj_, 'test_list')
    assert obj_.test_list == [4]

    obj_ = _setup_object()
    py2.append_to_attribute(obj_, 'test_list', [4], delist=True, prevent_duplicates=True)
    assert hasattr(obj_, 'test_list')
    assert obj_.test_list == [4]

    obj_ = _setup_object()
    py2.append_to_attribute(obj_, 'test_list', (4,), delist=True, prevent_duplicates=True)
    assert hasattr(obj_, 'test_list')
    assert obj_.test_list == [4]

    obj_ = _setup_object()
    py2.append_to_attribute(obj_, 'test_new', 5, delist=True, prevent_duplicates=True)
    assert hasattr(obj_, 'test_new')
    assert obj_.test_new == 5

    py2.append_to_attribute(obj_, 'test_new', 5, delist=True, prevent_duplicates=True)
    assert hasattr(obj_, 'test_new')
    assert obj_.test_new == 5

    py2.append_to_attribute(obj_, 'test_new', 6, delist=True, prevent_duplicates=True)
    assert hasattr(obj_, 'test_new')
    assert obj_.test_new == [5, 6]