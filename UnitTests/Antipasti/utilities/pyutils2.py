import Antipasti.utilities.pyutils2 as py2
import pytest


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


def test_autoname_layer_or_model():
    # Emulate a layer or a model with a dummy object
    layer_or_model = type('LayerOrModel', (object,), {})()

    # Cases with given layer
    name = py2.autoname_layer_or_model(layer_or_model)
    assert name == '{}_{}'.format('layerormodel', id(layer_or_model))

    name = py2.autoname_layer_or_model(layer_or_model, 'banana')
    assert name == '{}_{}'.format('banana', id(layer_or_model))

    with pytest.raises(AssertionError):
        name = py2.autoname_layer_or_model(layer_or_model, 0)

    # Cases without a given layer
    name = py2.autoname_layer_or_model(given_name='foo')
    assert name == 'foo'

    name = py2.autoname_layer_or_model(given_name='bar')
    assert name == 'bar'

    name = py2.autoname_layer_or_model(given_name='baz', force_postfix=True)
    assert name == 'baz_0'

    with pytest.raises(AssertionError):
        name = py2.autoname_layer_or_model()

    name = py2.autoname_layer_or_model(given_name='foo')
    assert name == 'foo_1'

    name = py2.autoname_layer_or_model(given_name='bar')
    assert name == 'bar_1'

    name = py2.autoname_layer_or_model(given_name='baz', force_postfix=True)
    assert name == 'baz_1'


if __name__ == '__main__':
    pytest.main([__file__])
